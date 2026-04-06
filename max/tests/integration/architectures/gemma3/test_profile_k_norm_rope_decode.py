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
from max.nn.kv_cache import unflatten_ragged_attention_inputs
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.core import TextContext
from torch.utils.dlpack import from_dlpack


PAGE_SIZE = 128
EPS = 1e-6
K_NORM_STD = 0.793
DEFAULT_WARMUP_ITERS = 20
DEFAULT_TIMED_ITERS = 50
CACHE_LEN_BASE = 1024
CACHE_LEN_STEP = 7
BATCH_SIZES = (64, 128)
KV_BLOCK_COMPARE_CHUNK_ELEMENTS = 8_388_608


def _load_text_config() -> dict[str, Any]:
    config_path = Path(os.environ["PIPELINES_TESTDATA"]) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    return data.get("text_config", data)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    return tuple(
        int(part.strip()) for part in value.split(",") if part.strip()
    )


def _profile_config(config: dict[str, Any]) -> dict[str, Any]:
    batch_sizes = _env_int_tuple(
        "PROFILE_K_NORM_ROPE_BATCH_SIZES", BATCH_SIZES
    )
    if not batch_sizes:
        raise ValueError(
            "PROFILE_K_NORM_ROPE_BATCH_SIZES must define at least one batch size"
        )

    warmup_iters = _env_int(
        "PROFILE_K_NORM_ROPE_WARMUP_ITERS", DEFAULT_WARMUP_ITERS
    )
    timed_iters = _env_int(
        "PROFILE_K_NORM_ROPE_TIMED_ITERS", DEFAULT_TIMED_ITERS
    )
    if warmup_iters < 0:
        raise ValueError("PROFILE_K_NORM_ROPE_WARMUP_ITERS must be >= 0")
    if timed_iters <= 0:
        raise ValueError("PROFILE_K_NORM_ROPE_TIMED_ITERS must be >= 1")

    return {
        "hidden_size": _env_int(
            "PROFILE_K_NORM_ROPE_HIDDEN_SIZE", config["hidden_size"]
        ),
        "num_attention_heads": _env_int(
            "PROFILE_K_NORM_ROPE_NUM_Q_HEADS", config["num_attention_heads"]
        ),
        "num_key_value_heads": _env_int(
            "PROFILE_K_NORM_ROPE_NUM_KV_HEADS",
            config["num_key_value_heads"],
        ),
        "head_dim": _env_int(
            "PROFILE_K_NORM_ROPE_HEAD_DIM", config["head_dim"]
        ),
        "rope_theta": _env_int(
            "PROFILE_K_NORM_ROPE_THETA", config["rope_local_base_freq"]
        ),
        "cache_len_base": _env_int(
            "PROFILE_K_NORM_ROPE_CACHE_LEN_BASE", CACHE_LEN_BASE
        ),
        "cache_len_step": _env_int(
            "PROFILE_K_NORM_ROPE_CACHE_LEN_STEP", CACHE_LEN_STEP
        ),
        "batch_sizes": batch_sizes,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "max_extra_steps": warmup_iters + timed_iters,
    }


def _make_weight_registry(head_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "k_gamma": torch.randn(head_dim, dtype=torch.bfloat16) * K_NORM_STD,
    }


def _make_text_context(length: int, max_length: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=max_length,
        tokens=TokenBuffer(np.zeros(length, dtype=np.int64)),
    )


def _build_graph(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    hidden_size: int,
    num_attention_heads: int,
    head_dim: int,
    rope_theta: int,
    batch_size: int,
    max_batch_size: int,
    cache_len_base: int,
    cache_len_step: int,
    max_extra_steps: int,
    use_fused: bool,
) -> Any:
    device_ref = DeviceRef.GPU()
    rope = Llama3RotaryEmbedding(
        hidden_size,
        num_attention_heads,
        rope_theta,
        cache_len_base + cache_len_step * (max_batch_size - 1) + max_extra_steps + 256,
        interleaved=False,
        head_dim=head_dim,
    )
    k_norm = Gemma3RMSNorm(head_dim, DType.bfloat16, EPS)
    k_norm.weight.name = "k_gamma"

    input_row_offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=device_ref,
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    graph_name = (
        f"Gemma3WideKNormRopeDecodeFusedBS{batch_size}"
        if use_fused
        else f"Gemma3WideKNormRopeDecodeBaselineBS{batch_size}"
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
        total_seq_len = Dim(batch_size)

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

    return session.load(graph, weights_registry=_make_weight_registry(head_dim))


def _make_runtime_inputs(
    *,
    session: InferenceSession,
    kv_params: KVCacheParams,
    batch_size: int,
    cache_len_base: int,
    cache_len_step: int,
    max_extra_steps: int,
) -> tuple[np.ndarray, Any]:
    cache_lengths = np.asarray(
        [
            cache_len_base + cache_len_step * request_idx
            for request_idx in range(batch_size)
        ],
        dtype=np.uint32,
    )
    max_cache_length = int(cache_lengths[-1]) + max_extra_steps + 1
    total_num_pages = sum(
        math.ceil((int(cache_length) + max_extra_steps + 1) / PAGE_SIZE)
        for cache_length in cache_lengths
    )

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=total_num_pages,
        session=session,
        max_batch_size=batch_size,
    )

    contexts: list[TextContext] = []
    for cache_length in cache_lengths:
        context = _make_text_context(int(cache_length), max_cache_length)
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(
            context,
            replica_idx=0,
            num_steps=max_extra_steps + 1,
        )
        contexts.append(context)

    runtime_inputs = kv_manager.runtime_inputs([contexts], num_steps=1).inputs[0]
    return cache_lengths, runtime_inputs


def _clone_kv_blocks(blocks: Buffer, seed: int) -> Buffer:
    torch.manual_seed(seed)
    shape = tuple(int(dim) for dim in blocks.shape)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
    return Buffer.from_dlpack(tensor)


def _device_uint32_buffer(array: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(array).to(device)


def _make_benchmark_args(
    *,
    batch_size: int,
    cache_lengths: np.ndarray,
    runtime_inputs: Any,
    device: Accelerator,
    blocks_seed: int,
    max_extra_steps: int,
) -> tuple[list[tuple[Any, ...]], Buffer]:
    row_offsets = _device_uint32_buffer(
        np.arange(batch_size + 1, dtype=np.uint32),
        device,
    )
    kv_blocks = _clone_kv_blocks(runtime_inputs.blocks, seed=blocks_seed)
    lookup_table = runtime_inputs.lookup_table.to(device)
    dispatch_metadata = runtime_inputs.attention_dispatch_metadata

    args: list[tuple[Any, ...]] = []
    for step in range(max_extra_steps + 1):
        step_cache_lengths = _device_uint32_buffer(
            cache_lengths + np.uint32(step), device
        )
        args.append(
            (
                row_offsets,
                kv_blocks,
                step_cache_lengths,
                lookup_table,
                runtime_inputs.max_lengths,
                dispatch_metadata,
            )
        )
    return args, kv_blocks


def _run_correctness_check(
    *,
    baseline: Any,
    fused: Any,
    baseline_args: tuple[Any, ...],
    fused_args: tuple[Any, ...],
) -> None:
    baseline_output = baseline.execute(*baseline_args)[0]
    fused_output = fused.execute(*fused_args)[0]
    torch.cuda.synchronize()

    torch.testing.assert_close(
        from_dlpack(baseline_output).to(torch.uint32),
        from_dlpack(fused_output).to(torch.uint32),
        rtol=0,
        atol=0,
    )

    baseline_blocks = from_dlpack(baseline_args[1]).to(torch.bfloat16)
    fused_blocks = from_dlpack(fused_args[1]).to(torch.bfloat16)
    flat_baseline_blocks = baseline_blocks.reshape(-1)
    flat_fused_blocks = fused_blocks.reshape(-1)
    for start in range(
        0, flat_baseline_blocks.numel(), KV_BLOCK_COMPARE_CHUNK_ELEMENTS
    ):
        end = start + KV_BLOCK_COMPARE_CHUNK_ELEMENTS
        torch.testing.assert_close(
            flat_baseline_blocks[start:end],
            flat_fused_blocks[start:end],
            rtol=2 * torch.finfo(torch.bfloat16).eps,
            atol=8 * torch.finfo(torch.bfloat16).eps,
        )


def _benchmark_us(
    compiled: Any,
    args: list[tuple[Any, ...]],
    *,
    warmup_iters: int,
    timed_iters: int,
) -> float:
    warmup_args = args[:warmup_iters]
    timed_args = args[warmup_iters : warmup_iters + timed_iters]

    for run_args in warmup_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()

    start_s = time.perf_counter()
    for run_args in timed_args:
        compiled.execute(*run_args)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    return elapsed_s * 1e6 / timed_iters


def test_profile_k_norm_rope_decode() -> None:
    config = _load_text_config()
    profile = _profile_config(config)
    session = InferenceSession(devices=[Accelerator(0)])
    device = Accelerator(0)
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        devices=[DeviceRef.GPU()],
        n_kv_heads=profile["num_key_value_heads"],
        head_dim=profile["head_dim"],
        num_layers=1,
        page_size=PAGE_SIZE,
    )

    results: dict[str, Any] = {
        "benchmark_config": {
            "dtype": "bfloat16",
            "hidden_size": profile["hidden_size"],
            "head_dim": profile["head_dim"],
            "num_q_heads": profile["num_attention_heads"],
            "num_kv_heads": profile["num_key_value_heads"],
            "page_size": PAGE_SIZE,
            "rope": "non-interleaved",
            "mode": "decode-ragged-live-graph-k-only",
            "cache_len_base": profile["cache_len_base"],
            "cache_len_step": profile["cache_len_step"],
            "batch_sizes": list(profile["batch_sizes"]),
            "warmup_iters": profile["warmup_iters"],
            "timed_iters": profile["timed_iters"],
        },
        "correctness": "pass",
        "first_sweep_us": {},
        "confirm_sweep_us": {},
        "average_us": {},
        "average_speedup_ratio_vs_graph_baseline": {},
    }

    max_batch_size = max(profile["batch_sizes"])
    for batch_size in profile["batch_sizes"]:
        cache_lengths, runtime_inputs = _make_runtime_inputs(
            session=session,
            kv_params=kv_params,
            batch_size=batch_size,
            cache_len_base=profile["cache_len_base"],
            cache_len_step=profile["cache_len_step"],
            max_extra_steps=profile["max_extra_steps"],
        )
        baseline = _build_graph(
            session=session,
            kv_params=kv_params,
            hidden_size=profile["hidden_size"],
            num_attention_heads=profile["num_attention_heads"],
            head_dim=profile["head_dim"],
            rope_theta=profile["rope_theta"],
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            cache_len_base=profile["cache_len_base"],
            cache_len_step=profile["cache_len_step"],
            max_extra_steps=profile["max_extra_steps"],
            use_fused=False,
        )
        fused = _build_graph(
            session=session,
            kv_params=kv_params,
            hidden_size=profile["hidden_size"],
            num_attention_heads=profile["num_attention_heads"],
            head_dim=profile["head_dim"],
            rope_theta=profile["rope_theta"],
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            cache_len_base=profile["cache_len_base"],
            cache_len_step=profile["cache_len_step"],
            max_extra_steps=profile["max_extra_steps"],
            use_fused=True,
        )
        run_name = (
            "decode_bs"
            f"{batch_size}_seq1_cache{profile['cache_len_base']}"
            f"_step{profile['cache_len_step']}_k_only"
        )

        correctness_baseline_args, correctness_baseline_blocks = (
            _make_benchmark_args(
                batch_size=batch_size,
                cache_lengths=cache_lengths,
                runtime_inputs=runtime_inputs,
                device=device,
                blocks_seed=100 + batch_size,
                max_extra_steps=profile["max_extra_steps"],
            )
        )
        correctness_fused_args, correctness_fused_blocks = _make_benchmark_args(
            batch_size=batch_size,
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=100 + batch_size,
            max_extra_steps=profile["max_extra_steps"],
        )
        _run_correctness_check(
            baseline=baseline,
            fused=fused,
            baseline_args=correctness_baseline_args[0],
            fused_args=correctness_fused_args[0],
        )
        del correctness_baseline_args
        del correctness_fused_args
        del correctness_baseline_blocks
        del correctness_fused_blocks

        first_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=200 + batch_size,
            max_extra_steps=profile["max_extra_steps"],
        )
        first_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=300 + batch_size,
            max_extra_steps=profile["max_extra_steps"],
        )
        first_baseline_us = _benchmark_us(
            baseline,
            first_baseline_args,
            warmup_iters=profile["warmup_iters"],
            timed_iters=profile["timed_iters"],
        )
        first_fused_us = _benchmark_us(
            fused,
            first_fused_args,
            warmup_iters=profile["warmup_iters"],
            timed_iters=profile["timed_iters"],
        )
        results["first_sweep_us"][run_name] = {
            "baseline": first_baseline_us,
            "fused": first_fused_us,
        }
        del first_baseline_args
        del first_fused_args

        confirm_baseline_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=400 + batch_size,
            max_extra_steps=profile["max_extra_steps"],
        )
        confirm_fused_args, _ = _make_benchmark_args(
            batch_size=batch_size,
            cache_lengths=cache_lengths,
            runtime_inputs=runtime_inputs,
            device=device,
            blocks_seed=500 + batch_size,
            max_extra_steps=profile["max_extra_steps"],
        )
        confirm_baseline_us = _benchmark_us(
            baseline,
            confirm_baseline_args,
            warmup_iters=profile["warmup_iters"],
            timed_iters=profile["timed_iters"],
        )
        confirm_fused_us = _benchmark_us(
            fused,
            confirm_fused_args,
            warmup_iters=profile["warmup_iters"],
            timed_iters=profile["timed_iters"],
        )
        results["confirm_sweep_us"][run_name] = {
            "baseline": confirm_baseline_us,
            "fused": confirm_fused_us,
        }
        del confirm_baseline_args
        del confirm_fused_args

        average_baseline_us = (first_baseline_us + confirm_baseline_us) / 2.0
        average_fused_us = (first_fused_us + confirm_fused_us) / 2.0
        results["average_us"][run_name] = {
            "baseline": average_baseline_us,
            "fused": average_fused_us,
        }
        results["average_speedup_ratio_vs_graph_baseline"][run_name] = (
            average_baseline_us / average_fused_us
        )

    speedups = list(results["average_speedup_ratio_vs_graph_baseline"].values())
    results["average_geomean_speedup_vs_graph_baseline"] = float(
        np.exp(np.mean(np.log(speedups)))
    )

    print("GEMMA3_WIDE_K_NORM_ROPE_DECODE_PROFILE_START")
    print(json.dumps(results, sort_keys=True))
    print("GEMMA3_WIDE_K_NORM_ROPE_DECODE_PROFILE_END")
