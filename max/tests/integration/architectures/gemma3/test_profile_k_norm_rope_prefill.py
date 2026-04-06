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

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, ops
from max.interfaces import RequestID, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn.kernels import (
    KVCacheParams,
    k_rms_norm_rope_ragged,
    rms_norm_key_cache,
    rope_k_cache_ragged,
)
from max.nn.kv_cache import KVCacheInputsPerDevice, unflatten_ragged_attention_inputs
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack


PAGE_SIZE = 128
EPS = 1e-6
K_NORM_STD = 0.793
WARMUP_ITERS = 20
TIMED_ITERS = 50
PREFILL_SHAPES = (
    (1, 11),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (2, 2048),
)
PLACEHOLDER_CACHE_LEN = 1


def _resolve_prefill_shapes() -> tuple[tuple[int, int], ...]:
    raw_shapes = os.environ.get("PROFILE_K_NORM_ROPE_PREFILL_SHAPES", "").strip()
    if raw_shapes == "":
        return PREFILL_SHAPES

    resolved: list[tuple[int, int]] = []
    for raw_shape in raw_shapes.split(","):
        shape_text = raw_shape.strip().lower()
        if shape_text == "":
            continue
        if "x" not in shape_text:
            raise ValueError(
                "PROFILE_K_NORM_ROPE_PREFILL_SHAPES entries must look like "
                f"'batchxseq', got {raw_shape!r}"
            )
        batch_text, seq_text = shape_text.split("x", maxsplit=1)
        shape = (int(batch_text), int(seq_text))
        if shape not in PREFILL_SHAPES:
            raise ValueError(
                "PROFILE_K_NORM_ROPE_PREFILL_SHAPES only supports the "
                f"existing prefill grid {PREFILL_SHAPES}, got {shape}"
            )
        if shape not in resolved:
            resolved.append(shape)

    if not resolved:
        raise ValueError(
            "PROFILE_K_NORM_ROPE_PREFILL_SHAPES must select at least one shape"
        )
    return tuple(resolved)


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _make_weight_registry(head_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "k_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * K_NORM_STD,
    }


def _make_placeholder_text_context(max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(
            np.zeros(PLACEHOLDER_CACHE_LEN, dtype=np.int64)
        ),
    )


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _row_offsets_array(batch_size: int, seq_len: int) -> np.ndarray:
    return np.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=np.uint32,
    )


def _prefill_run_name(batch_size: int, seq_len: int) -> str:
    return f"prefill_bs{batch_size}_seq{seq_len}_cache0_step0_k_only"


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    config: dict[str, Any],
    weight_registry: dict[str, torch.Tensor],
    batch_size: int,
    seq_len: int,
    use_fused: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    total_seq_len = Dim(batch_size * seq_len)
    rope = Llama3RotaryEmbedding(
        config["hidden_size"],
        config["num_attention_heads"],
        config["rope_local_base_freq"],
        seq_len + 256,
        interleaved=False,
        head_dim=config["head_dim"],
    )
    k_norm = Gemma3RMSNorm(config["head_dim"], DType.bfloat16, EPS)
    k_norm.weight.name = "k_gamma"

    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = (
        f"Gemma3KOnlyNormRopePrefillFusedBS{batch_size}Seq{seq_len}"
        if use_fused
        else f"Gemma3KOnlyNormRopePrefillBaselineBS{batch_size}Seq{seq_len}"
    )

    with Graph(
        graph_name,
        input_types=(input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        freqs_cis = ops.cast(rope.freqs_cis, DType.bfloat16).to(device_ref)
        gamma = k_norm.weight.cast(kv_params.dtype).to(device_ref)

        if use_fused:
            k_rms_norm_rope_ragged(
                kv_params,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets.tensor,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                gamma=gamma,
                epsilon=EPS,
                layer_idx=layer_idx,
                weight_offset=1.0,
                interleaved=False,
            )
        else:
            rms_norm_key_cache(
                kv_params,
                kv_collection=kv_collection,
                gamma=gamma,
                epsilon=EPS,
                layer_idx=layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets.tensor,
                weight_offset=1.0,
            )
            rope_k_cache_ragged(
                kv_params,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets.tensor,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                layer_idx=layer_idx,
                interleaved=False,
            )

        graph.output(input_row_offsets.tensor)

    return session.load(graph, weights_registry=weight_registry)


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    seq_len: int,
    device: Accelerator,
) -> KVCacheInputsPerDevice:
    max_cache_length = PLACEHOLDER_CACHE_LEN + seq_len
    total_num_pages = batch_size * math.ceil(max_cache_length / PAGE_SIZE)
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for _ in range(batch_size):
        context = _make_placeholder_text_context(max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=seq_len,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs(
        [contexts],
        num_steps=seq_len,
    ).inputs[0]

    return KVCacheInputsPerDevice(
        blocks=runtime_inputs.blocks,
        cache_lengths=_device_uint32_buffer(
            np.zeros(batch_size, dtype=np.uint32), device
        ),
        lookup_table=runtime_inputs.lookup_table,
        max_lengths=runtime_inputs.max_lengths,
        kv_scales=runtime_inputs.kv_scales,
        attention_dispatch_metadata=runtime_inputs.attention_dispatch_metadata,
    )


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    return Buffer.from_dlpack(tensor)


def _make_benchmark_args(
    *,
    batch_size: int,
    seq_len: int,
    runtime_inputs: KVCacheInputsPerDevice,
    device: Accelerator,
    blocks_seed: int,
) -> tuple[Any, ...]:
    row_offsets = _device_uint32_buffer(
        _row_offsets_array(batch_size, seq_len),
        device,
    )
    return (
        row_offsets,
        _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed),
        runtime_inputs.cache_lengths,
        runtime_inputs.lookup_table,
        runtime_inputs.max_lengths,
        runtime_inputs.attention_dispatch_metadata,
    )


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    baseline_args: tuple[Any, ...],
    fused_args: tuple[Any, ...],
) -> None:
    baseline.execute(*baseline_args)
    fused.execute(*fused_args)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        from_dlpack(baseline_args[1]).to(torch.bfloat16),
        from_dlpack(fused_args[1]).to(torch.bfloat16),
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def _benchmark_us(compiled: Any, args: list[tuple[Any, ...]]) -> float:
    warmup_args = args[:WARMUP_ITERS]
    timed_args = args[WARMUP_ITERS : WARMUP_ITERS + TIMED_ITERS]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / len(timed_args)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def test_profile_k_norm_rope_prefill() -> None:
    config = _load_text_config()
    prefill_shapes = _resolve_prefill_shapes()
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        num_layers=1,
        page_size=PAGE_SIZE,
    )
    weight_registry = _make_weight_registry(config["head_dim"])

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "head_dim": config["head_dim"],
            "num_kv_heads": config["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "rope": "non-interleaved",
            "mode": "prefill-ragged-live-graph-k-only",
            "shapes": [
                _prefill_run_name(batch_size, seq_len)
                for batch_size, seq_len in prefill_shapes
            ],
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "placeholder_cache_len": PLACEHOLDER_CACHE_LEN,
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_graph_baseline": {},
    }

    for batch_size, seq_len in prefill_shapes:
        run_name = _prefill_run_name(batch_size, seq_len)
        runtime_inputs = _make_runtime_inputs(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        baseline = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=False,
        )
        fused = _build_graph(
            session=session,
            kv_params=kv_params,
            config=config,
            weight_registry=weight_registry,
            batch_size=batch_size,
            seq_len=seq_len,
            use_fused=True,
        )

        baseline_correctness_args = _make_benchmark_args(
            batch_size=batch_size,
            seq_len=seq_len,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=1000 + batch_size + seq_len,
        )
        fused_correctness_args = _make_benchmark_args(
            batch_size=batch_size,
            seq_len=seq_len,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=1000 + batch_size + seq_len,
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=baseline_correctness_args,
            fused_args=fused_correctness_args,
        )

        first_baseline_args = [
            _make_benchmark_args(
                batch_size=batch_size,
                seq_len=seq_len,
                runtime_inputs=runtime_inputs,
                device=device,
                blocks_seed=2000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_fused_args = [
            _make_benchmark_args(
                batch_size=batch_size,
                seq_len=seq_len,
                runtime_inputs=runtime_inputs,
                device=device,
                blocks_seed=2000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        first_baseline_us = _benchmark_us(baseline, first_baseline_args)
        first_fused_us = _benchmark_us(fused, first_fused_args)
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }

        confirm_baseline_args = [
            _make_benchmark_args(
                batch_size=batch_size,
                seq_len=seq_len,
                runtime_inputs=runtime_inputs,
                device=device,
                blocks_seed=3000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_fused_args = [
            _make_benchmark_args(
                batch_size=batch_size,
                seq_len=seq_len,
                runtime_inputs=runtime_inputs,
                device=device,
                blocks_seed=3000 + batch_size + seq_len + iteration,
            )
            for iteration in range(WARMUP_ITERS + TIMED_ITERS)
        ]
        confirm_baseline_us = _benchmark_us(baseline, confirm_baseline_args)
        confirm_fused_us = _benchmark_us(fused, confirm_fused_args)
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_graph_baseline"][run_name] = (
            average_baseline_us / average_fused_us
        )

    ratios = list(results["average_speedup_ratio_vs_graph_baseline"].values())
    results["average_geomean_speedup_vs_graph_baseline"] = _geomean(ratios)

    print(json.dumps(results, indent=2, sort_keys=True))
