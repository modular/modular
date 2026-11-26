# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Benchmarks the Qwen3-VL text encoder (Ideogram 4 tap set) by path.

The model is the one Ideogram 4 conditions on: Qwen3-VL-8B's text tower
(36 layers, 13 layer taps, 53248-wide concat) run over prompt tokens only.
Weights are random, so the timings are shape-accurate without a checkpoint.

Three paths are timed per sequence length:

``forward``  the compiled graph on a prebuilt device mask -- the GPU work.
``host``     what ``Qwen3TextEncoderModel.__call__`` does per request: build
             the additive causal mask on the CPU, copy it up, then run.
``graph``    the same input captured as a device graph and replayed.

Knobs come from the environment so ``bazel test`` can set them::

    TE_SEQ_LENS=512,1024 TE_ITERS=20 TE_ATTN=causal bazelw test -c opt \
        //max/tests/integration/architectures/qwen3_modulev3:test_profile_text_encoder \
        --test_env=TE_SEQ_LENS --test_env=TE_ITERS --test_env=TE_ATTN

``TE_ATTN=causal`` builds the graph whose attention derives the causal mask
in the kernel; the default ``additive`` builds the materialized-mask graph
production ran. ``TE_MASK_ONLY=1`` times the host mask build alone.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.qwen3_modulev3.text_encoder.model_config import (
    Qwen3TextEncoderConfig,
)
from max.pipelines.architectures.qwen3_modulev3.text_encoder.qwen3 import (
    Qwen3TextEncoderTransformer,
)
from max.pipelines.modeling.dataprocessing.causal_attention_mask import (
    causal_attention_mask_with_token_mask,
)
from max.tensor import Tensor, compilation
from max.tensor.sharding import TensorLayout

# Qwen3-VL layers whose hidden states Ideogram 4 concatenates (HF indices,
# shifted by one for the hidden-states contract: index i is the output of
# block i - 1). Must match ideogram4.model_config.QWEN3_VL_ACTIVATION_LAYERS.
IDEogram_TAPS: tuple[int, ...] = tuple(
    layer + 1 for layer in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
)


def _config(max_seq_len: int) -> Qwen3TextEncoderConfig:
    """Qwen3-VL-8B's text_config with the Ideogram tap set."""
    return Qwen3TextEncoderConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=36,
        head_dim=128,
        vocab_size=151936,
        intermediate_size=12288,
        # Qwen3-VL stores rope_theta under rope_parameters (5e6).
        rope_theta=5000000.0,
        max_seq_len=max_seq_len,
        rms_norm_eps=1e-6,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
        hidden_state_layers=list(IDEogram_TAPS),
    )


def _build(
    max_seq_len: int, device: Accelerator, use_attention_bias: bool = True
) -> Any:
    """Compiles the encoder with zero-filled weights of the right shapes."""
    config = _config(max_seq_len)
    with F.lazy():
        model = Qwen3TextEncoderTransformer(
            config, use_attention_bias=use_attention_bias
        )
        model.to(device)

    weights: dict[str, Tensor] = {}
    for name, param in model.parameters:
        shape = tuple(int(d) for d in param.shape)
        weights[name] = Tensor(storage=Buffer(param.dtype, shape))
    n_elements = sum(
        int(np.prod(tuple(int(d) for d in w.shape))) for w in weights.values()
    )
    print(
        f"weights: {len(weights)} tensors, {n_elements / 2**30:.2f} Gelements"
    )
    return model.compile(*model.input_types(), weights=weights)


def _mask_numpy(seq_len: int, device: Accelerator) -> Tensor:
    """Production's per-request additive mask, built on the CPU."""
    additive = causal_attention_mask_with_token_mask(
        [0], np.ones((seq_len,), dtype=np.bool_)
    )
    return Tensor(
        storage=Buffer.from_numpy(additive[:, np.newaxis, :, :]).to(device)
    )


def _time(
    fn: Callable[[], object],
    device: Accelerator,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Median/p10 microseconds of ``fn`` with the device synced per sample."""
    for _ in range(warmup):
        fn()
    device.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        device.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return {
        "p50_us": statistics.median(samples),
        "p10_us": sorted(samples)[max(0, len(samples) // 10)],
    }


def _time_concat(
    device: Accelerator, seq_len: int, iters: int, warmup: int
) -> None:
    """Times the 13-tap concat chain exactly as the pipeline runs it.

    The encoder stacks the taps layer-major and the Ideogram pipeline then
    permutes them hidden-major, so one request pays two full transposes of
    ``[S, 13, 4096]``. This prices that chain alone.
    """
    hidden = 4096
    taps = len(IDEogram_TAPS)
    dtype = DType.bfloat16

    def concat(*xs: Tensor) -> Tensor:
        stacked = F.stack(list(xs), axis=0)  # [L, S, D]
        stacked = F.unsqueeze(stacked, 0)  # [1, L, S, D]
        stacked = F.permute(stacked, [0, 2, 1, 3])  # [1, S, L, D]
        return F.reshape(stacked, [1, seq_len, taps * hidden])

    def reorder(x: Tensor) -> Tensor:
        x = F.reshape(x, (1, seq_len, taps, hidden))
        x = F.permute(x, (0, 1, 3, 2))
        return F.reshape(x, (1, seq_len, hidden * taps))

    layouts = [TensorLayout(dtype, [seq_len, hidden], device)] * taps
    run_concat = compilation.compile(concat)(*layouts)
    run_reorder = compilation.compile(reorder)(
        TensorLayout(dtype, [1, seq_len, taps * hidden], device)
    )
    xs = [
        Tensor.zeros([seq_len, hidden], dtype=dtype, device=device)
        for _ in range(taps)
    ]
    mid = Tensor.zeros([1, seq_len, taps * hidden], dtype=dtype, device=device)

    for name, fn in (
        ("concat", lambda: run_concat(*xs)),
        ("reorder", lambda: run_reorder(mid)),
    ):
        stats = _time(fn, device, warmup, iters)
        print(
            f"{name:8s} seq={seq_len:5d}  p50={stats['p50_us']:9.1f} us  "
            f"p10={stats['p10_us']:9.1f} us"
        )


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    return [int(v) for v in raw.split(",")] if raw else default


def test_profile_text_encoder() -> None:
    """Entry point for `bazel test`; knobs come from the environment."""
    run(
        seq_lens=_env_int_list("TE_SEQ_LENS", [512, 1024]),
        iters=_env_int("TE_ITERS", 20),
        warmup=_env_int("TE_WARMUP", 5),
        mask_only=os.environ.get("TE_MASK_ONLY", "0") == "1",
    )


def run(
    seq_lens: list[int], iters: int, warmup: int, mask_only: bool = False
) -> None:
    args = argparse.Namespace(
        seq_lens=seq_lens, iters=iters, warmup=warmup, mask_only=mask_only
    )
    causal = os.environ.get("TE_ATTN", "additive") == "causal"
    device = Accelerator()
    if os.environ.get("TE_CONCAT", "0") == "1":
        for seq_len in args.seq_lens:
            _time_concat(device, seq_len, args.iters, args.warmup)
        return
    if args.mask_only:
        for seq_len in args.seq_lens:
            stats = _time(
                partial(_mask_numpy, seq_len, device),
                device,
                args.warmup,
                args.iters,
            )
            print(
                f"mask seq={seq_len:5d}  p50={stats['p50_us']:9.1f} us  "
                f"p10={stats['p10_us']:9.1f} us"
            )
        return

    compiled = _build(max(args.seq_lens), device, not causal)
    print(
        f"attention mask: {'causal (no mask input)' if causal else 'additive'}"
    )

    for seq_len in args.seq_lens:
        rng = np.random.default_rng(0)
        tokens_np = rng.integers(0, 151936, size=(seq_len,), dtype=np.int64)
        tokens = Tensor(tokens_np, device=device)
        bias = None if causal else _mask_numpy(seq_len, device)
        device.synchronize()

        results: dict[str, dict[str, float]] = {}
        if causal:
            results["forward"] = _time(
                partial(compiled, tokens),
                device,
                args.warmup,
                args.iters,
            )
        else:
            results["forward"] = _time(
                partial(compiled, tokens, bias),
                device,
                args.warmup,
                args.iters,
            )

            def host_call(t: Tensor = tokens, s: int = seq_len) -> Any:
                """What the model's __call__ does per request."""
                return compiled(t, _mask_numpy(s, device))

            results["host"] = _time(host_call, device, args.warmup, args.iters)

        # Device graph: capture once per shape, then replay.
        engine = compiled.engine_model
        key = seq_len
        token_buffer = tokens.storage
        assert token_buffer is not None
        buffers: list[Buffer] = [token_buffer]
        if bias is not None:
            bias_buffer = bias.storage
            assert bias_buffer is not None
            buffers.append(bias_buffer)
        inputs = compiled.engine_inputs(buffers)
        outs = engine.capture(key, *inputs)
        device.synchronize()
        results["graph"] = _time(
            partial(engine.replay, key, *inputs),
            device,
            args.warmup,
            args.iters,
        )
        del outs
        engine.release_captured_graph(key)

        for mode, stats in results.items():
            print(
                f"{mode:8s} seq={seq_len:5d}  p50={stats['p50_us']:9.1f} us  "
                f"p10={stats['p10_us']:9.1f} us"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mask-only", action="store_true")
    cli = parser.parse_args()
    run(cli.seq_lens, cli.iters, cli.warmup, cli.mask_only)
