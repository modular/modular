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

# Benchmark MLA (Multi-head Latent Attention) decode kernels for DeepSeek models.
# Compares FlashInfer's TRT-LLM MLA implementation against MAX's MLA implementation.
# Run via kbench: kbench bench_mla_decode.yaml

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from dataclasses import dataclass
from typing import Any

import torch

# Import bench utilities from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# MAX imports
from bench import bench_kineto_with_cupti_warmup, setup_ninja_path
from bencher_utils import Bench, ThroughputMeasure
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kernels import flare_mla_decode_ragged
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedCacheValues,
)

LINE = "=" * 80


def to_float8(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 format.

    Args:
        x: Input tensor to quantize
        dtype: Target FP8 dtype (default: float8_e4m3fn)

    Returns:
        Tuple of (quantized_tensor, scale) where scale is the dequantization scale
    """
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None


@dataclass
class Config:
    num_q_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int


def calculate_mla_memory_bytes(
    batch_size: int,
    q_len_per_request: int,
    cache_len: int,
    dtype: torch.dtype,
    model_config: Config,
) -> int:
    """Calculate memory throughput for MLA operations.

    Args:
        batch_size: Number of sequences
        q_len_per_request: Query length per request
        cache_len: KV cache length per sequence
        dtype: Data type of tensors
        time_s: Execution time in seconds

    Returns:
        Memory throughput in GB/s
    """
    num_q_heads = model_config.num_q_heads
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim

    # MLA reads compressed KV cache and query, writes output
    # Read: query + kv_cache (in input dtype)
    # Write: output (always bf16 for FP8 inputs, otherwise same as input dtype)
    if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        input_bytes_per_element = 1
        # FP8 MLA kernels produce bf16 output (see bmm2_scale comment in bench_flashinfer_trtllm)
        output_bytes_per_element = 2
    elif dtype in [torch.bfloat16, torch.float16]:
        input_bytes_per_element = 2
        output_bytes_per_element = 2
    else:
        input_bytes_per_element = 4
        output_bytes_per_element = 4

    # Input: query [batch_size, q_len_per_request, num_q_heads, kv_lora_rank + qk_rope_head_dim]
    query_bytes = (
        batch_size
        * q_len_per_request
        * num_q_heads
        * (kv_lora_rank + qk_rope_head_dim)
        * input_bytes_per_element
    )

    # KV cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim] (read only what's needed)
    kv_bytes = (
        batch_size
        * cache_len
        * (kv_lora_rank + qk_rope_head_dim)
        * input_bytes_per_element
    )

    # Output: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank]
    output_bytes = (
        batch_size
        * q_len_per_request
        * num_q_heads
        * kv_lora_rank
        * output_bytes_per_element
    )

    total_bytes = query_bytes + kv_bytes + output_bytes
    return total_bytes


def bench_flashinfer_trtllm(
    batch_size: int,
    cache_len: int,
    page_size: int,
    dtype: torch.dtype,
    model_config: Config,
    q_len_per_request: int = 1,
    enable_pdl: bool | None = None,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    """Benchmark FlashInfer MLA decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence (max sequence length)
        page_size: Page/block size for paged KV cache
        dtype: torch dtype for inputs (supports bfloat16, float16, float8_e4m3fn)
        q_len_per_request: Query length per request (1 for decode, >1 for chunked prefill)
        enable_pdl: Enable PDL (Persistent Dynamic Load) optimization
    """
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    device = "cuda"
    is_fp8 = dtype == torch.float8_e4m3fn

    # DeepSeek MLA configuration
    num_q_heads = model_config.num_q_heads
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_lora_rank = model_config.kv_lora_rank

    # For FP8: generate in bf16 first, then quantize for proper scaling
    gen_dtype = torch.bfloat16 if is_fp8 else dtype

    # Query tensor: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank + qk_rope_head_dim]
    query_raw = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=gen_dtype,
        device=device,
    )

    # Sequence lengths - all sequences have cache_len for simplicity
    seq_lens_tensor = torch.full(
        (batch_size,), cache_len, dtype=torch.int, device=device
    )

    # Calculate number of blocks per sequence
    max_num_blocks_per_seq = (cache_len + page_size - 1) // page_size

    # Generate block tables using arange (simple sequential block IDs)
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Calculate total number of blocks needed for KV cache allocation
    num_blocks = batch_size * max_num_blocks_per_seq

    # Create MLA KV cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
    # This is the compressed format for MLA - stores ckv (compressed kv) and kpe (key positional encoding)
    kv_cache_raw = torch.randn(
        num_blocks,
        page_size,
        kv_lora_rank + qk_rope_head_dim,
        dtype=gen_dtype,
        device=device,
    )

    # For FP8: quantize and compute proper scales
    # bmm1_scale = q_scale * k_scale * sm_scale (where sm_scale = 1/sqrt(head_dim))
    # bmm2_scale = v_scale * o_scale (typically 1.0 for bf16 output)
    if is_fp8:
        query, q_scale = to_float8(query_raw)
        kv_cache, kv_scale = to_float8(kv_cache_raw)
        q_scale_val = q_scale.item()
        kv_scale_val = kv_scale.item()
        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        bmm1_scale = q_scale_val * kv_scale_val * sm_scale
        bmm2_scale = (
            kv_scale_val  # v_scale * o_scale (o_scale=1 for bf16 output)
        )
    else:
        query = query_raw
        kv_cache = kv_cache_raw
        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        bmm1_scale = sm_scale
        bmm2_scale = 1.0

    # Workspace buffer (must be zero-initialized for trtllm-gen backend)
    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    def run_kernel() -> torch.Tensor:
        return _flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(
                1
            ),  # Add layer dimension: [num_blocks, 1, page_size, ...]
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=cache_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            enable_pdl=enable_pdl,
            backend="trtllm-gen",
        )

    # Warmup
    for _ in range(10):
        run_kernel()

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
    )

    # If running under external profilers (ncu/nsys), skip kineto to avoid assertion failures.
    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return 1.0, total_bytes

    # Benchmark with CUPTI warmup for CUTLASS kernels
    try:
        time_s = bench_kineto_with_cupti_warmup(
            run_kernel,
            kernel_names="fmhaSm100",  # FlashInfer TRT-LLM MLA kernel name prefix
            num_tests=100,
            suppress_kineto_output=True,
            flush_l2=True,
        )
        assert isinstance(time_s, float)  # Single kernel_name returns float
    except RuntimeError as e:
        # If kineto fails (e.g., running under ncu/nsys), return dummy time
        if "No kernel times found" in str(e):
            print(
                f"Warning: kineto profiling failed (likely running under ncu/nsys). "
                f"Use --no-kineto flag to skip kineto. Error: {e}"
            )
            run_kernel()
            torch.cuda.synchronize()
            return 1.0, total_bytes
        raise

    return time_s, total_bytes


def bench_max(
    batch_size: int,
    cache_len: int,
    page_size: int,
    dtype: torch.dtype,
    model_config: Config,
    q_len_per_request: int = 1,
    no_kineto: bool = False,
) -> tuple[float, int]:
    """Benchmark MAX MLA decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        page_size: Page size for paged KV cache
        dtype: torch dtype for inputs
        q_len_per_request: Query length per request (1 for decode)
    """
    # Convert torch dtype to MAX DType
    max_dtype = DType.from_torch(dtype)

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # DeepSeek MLA configuration
    num_q_heads = model_config.num_q_heads
    num_kv_heads = 1  # MLA uses 1 KV head (MQA with compression)
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_lora_rank = model_config.kv_lora_rank
    qk_head_dim = kv_lora_rank + qk_rope_head_dim  # Total query/key dimension

    # Setup KV cache configuration for MLA
    # MLA stores compressed KV (kv_lora_rank) + rope embeddings (qk_rope_head_dim)
    kv_params = KVCacheParams(
        dtype=max_dtype,
        n_kv_heads=num_kv_heads,
        head_dim=kv_lora_rank + qk_rope_head_dim,  # Compressed dimension
        num_layers=1,  # Benchmarking a single layer
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        is_mla=True,
    )

    num_blocks_per_seq = (cache_len + page_size - 1) // page_size

    # For MLA: [num_pages, 1 (K and V compressed together), 1 layer, page_size, 1 kv_head, compressed_dim]
    # MLA stores compressed KV cache, not separate K and V
    paged_blocks_torch = torch.randn(
        batch_size * num_blocks_per_seq,
        1,  # MLA uses compressed KV format, not separate K and V
        1,
        page_size,
        num_kv_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=dtype,
        device="cuda",
    )

    lut_torch = (
        torch.arange(
            batch_size * num_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        .reshape(batch_size, num_blocks_per_seq)
        .to(torch.uint32)
    )

    cache_lengths_torch = torch.full(
        (batch_size,), cache_len, dtype=torch.uint32, device="cuda"
    )

    # For decode: max_seq_length=q_len_per_request, max_cache_length=cache_len
    max_lengths_torch = torch.tensor(
        [[q_len_per_request, cache_len]], dtype=torch.uint32, device="cpu"
    )

    # Convert torch tensors to MAX types
    paged_blocks_max = Buffer.from_dlpack(paged_blocks_torch)
    lut_max = Buffer.from_dlpack(lut_torch)
    cache_lengths_max = Buffer.from_dlpack(cache_lengths_torch)
    max_lengths_max = Buffer.from_dlpack(max_lengths_torch)

    # Define input types
    # Query for MLA decode: [total_tokens, num_q_heads, qk_head_dim]
    # where qk_head_dim = kv_lora_rank + qk_rope_head_dim
    q_type = TensorType(
        max_dtype,
        shape=["total_tokens", num_q_heads, qk_head_dim],
        device=DeviceRef.GPU(),
    )

    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["batch_size_plus_1"],
        device=DeviceRef.GPU(),
    )

    blocks_type = BufferType(
        max_dtype,
        shape=[
            "total_num_pages",
            1,
            1,
            page_size,
            num_kv_heads,
            kv_lora_rank + qk_rope_head_dim,
        ],
        device=DeviceRef.GPU(),
    )

    cache_lengths_type = TensorType(
        DType.uint32,
        shape=["batch_size"],
        device=DeviceRef.GPU(),
    )

    lookup_table_type = TensorType(
        DType.uint32,
        shape=["batch_size", "max_num_pages"],
        device=DeviceRef.GPU(),
    )

    max_lengths_type = TensorType(
        DType.uint32,
        shape=[1, 2],
        device=DeviceRef.CPU(),
    )

    # Build graph with MLA decode
    with Graph(
        "mla_decode_max",
        input_types=[
            q_type,
            input_row_offsets_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_lengths_type,
        ],
    ) as graph:
        (
            q,
            input_row_offsets,
            blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
        ) = graph.inputs

        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_lengths.tensor,
        )

        # Use MLA decode kernel
        result = flare_mla_decode_ragged(
            kv_params,
            q.tensor,
            input_row_offsets.tensor,
            kv_collection,
            layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim),
            qk_rope_dim=qk_rope_head_dim,
        )

        graph.output(result)

    # Compile model
    model = session.load(graph)

    # Prepare inputs
    # Query: [batch_size * q_len_per_request, num_q_heads, qk_head_dim]
    total_tokens = batch_size * q_len_per_request
    q_input = torch.randn(
        total_tokens, num_q_heads, qk_head_dim, dtype=dtype, device="cuda"
    )

    # Input row offsets for ragged tensor
    input_row_offsets = torch.arange(
        0, total_tokens + 1, q_len_per_request, dtype=torch.int32, device="cuda"
    ).to(torch.uint32)

    def run_kernel() -> Any:
        output = model.execute(
            q_input.detach(),
            input_row_offsets.detach(),
            paged_blocks_max,
            cache_lengths_max,
            lut_max,
            max_lengths_max,
        )[0]
        return output

    # Warmup
    for _ in range(10):
        run_kernel()

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
    )

    # If running under external profilers (ncu/nsys), skip kineto to avoid assertion failures.
    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return 1.0, total_bytes

    # Benchmark with CUPTI warmup
    # Note: Split-K implementation uses two kernels (decode + combine), so we
    # need with_multiple_kernels=True to sum their times
    try:
        time_s = bench_kineto_with_cupti_warmup(
            run_kernel,
            kernel_names="mla",
            num_tests=100,
            suppress_kineto_output=True,
            flush_l2=True,
            with_multiple_kernels=True,
        )
        assert isinstance(time_s, float)  # Single kernel_name returns float
    except RuntimeError as e:
        # If kineto fails (e.g., running under ncu/nsys), return dummy time
        if "No kernel times found" in str(e):
            print(
                f"Warning: kineto profiling failed (likely running under ncu/nsys). "
                f"Use --no-kineto flag to skip kineto. Error: {e}"
            )
            run_kernel()
            torch.cuda.synchronize()
            return 1.0, total_bytes
        raise

    return time_s, total_bytes


def bench_mla_decode(
    batch_size: int,
    cache_len: int,
    dtype: torch.dtype,
    model_config: Config,
    engine: str,  # "modular_max" or "flashinfer"
    q_len_per_request: int = 1,
    backend: str = "trtllm-gen",
    enable_pdl: bool | None = None,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    """Run all MLA decode benchmarks and return results.

    Args:
        batch_size: Batch size (number of sequences)
        cache_len: KV cache length per sequence (max sequence length)
        dtype: torch dtype for inputs (e.g., torch.bfloat16, torch.float8_e4m3fn)
        q_len_per_request: Query length per request (1 for decode, >1 for chunked prefill)
        backend: Backend for FlashInfer ("trtllm-gen" or "xqa")
        enable_pdl: Enable PDL optimization for FlashInfer

    Returns:
        Dictionary with benchmark results: {"flashinfer": (time_s, gb_per_sec), "max": (time_s, gb_per_sec)}
    """
    # Use appropriate page sizes for each backend
    flashinfer_page_size = 64  # FlashInfer TRT-LLM tested with 32 and 64
    max_page_size = 128  # MAX only supports 128

    result: tuple[float, int] | None = None
    if engine == "flashinfer":
        # Run FlashInfer benchmark with TensorRT-LLM MLA backend
        if _flashinfer is not None:
            try:
                result = bench_flashinfer_trtllm(
                    batch_size=batch_size,
                    cache_len=cache_len,
                    page_size=flashinfer_page_size,
                    dtype=dtype,
                    model_config=model_config,
                    q_len_per_request=q_len_per_request,
                    enable_pdl=enable_pdl,
                    no_kineto=no_kineto,
                )
            except Exception as e:
                print(f"FlashInfer benchmark failed: {e}")
                import traceback

                traceback.print_exc()

    # Run MAX benchmark
    if engine == "modular_max":
        try:
            result = bench_max(
                batch_size=batch_size,
                cache_len=cache_len,
                page_size=max_page_size,
                dtype=dtype,
                model_config=model_config,
                q_len_per_request=q_len_per_request,
                no_kineto=no_kineto,
            )
        except Exception as e:
            print(f"MAX benchmark failed: {e}")
            import traceback

            traceback.print_exc()
    return result


if __name__ == "__main__":
    # DeepSeek MLA configuration (fixed for DeepSeek V2/V3)
    NUM_Q_HEADS = 128
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    KV_LORA_RANK = 512

    cfg = Config(NUM_Q_HEADS, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, KV_LORA_RANK)

    parser = argparse.ArgumentParser(description="MHA Decode Benchmark")
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=128, help="Batch size"
    )

    parser.add_argument(
        "--cache_len",
        "--cache-len",
        type=int,
        default=1024,
        help="KV cache length",
    )
    parser.add_argument(
        "--q_len_per_request",
        "--q-len-per-request",
        type=int,
        default=1,
        help="Q Length per Request",
    )
    parser.add_argument(
        "--num_q_heads",
        "--num-q-heads",
        type=int,
        default=cfg.num_q_heads,
        help="Number of query heads",
    )

    parser.add_argument(
        "--qk_nope_head_dim",
        "--qk-nope-head-dim",
        type=int,
        default=cfg.qk_nope_head_dim,
        help="qk nope head dim",
    )

    parser.add_argument(
        "--qk_rope_head_dim",
        "--qk-rope-head-dim",
        type=int,
        default=cfg.qk_rope_head_dim,
        help="qk rope head dim",
    )

    parser.add_argument(
        "--kv_lora_rank",
        "--kv-lora-rank",
        type=int,
        default=cfg.qk_rope_head_dim,
        help="kv lora rank",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "float8_e4m3fn"],
        help="Data type (float8_e4m3fn for FP8 quantized MLA)",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=["modular_max", "flashinfer"],
        help="Engine",
    )
    parser.add_argument(
        "--no-kineto",
        action="store_true",
        help="Skip kineto timing (for ncu/nsys).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )
    args, _ = parser.parse_known_args()

    # TODO: overlap "engine" with "backend"
    if args.engine not in ["flashinfer", "modular_max"]:
        raise ValueError(f"engine {args.engine} is not supported!")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    result = bench_mla_decode(
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
        model_config=cfg,
        q_len_per_request=args.q_len_per_request,
        backend="trtllm-gen",
        enable_pdl=True,
        no_kineto=args.no_kineto,
    )

    met_sec, bytes = result if result else [0, 0]
    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    name = (
        f"MLA_Decode/batch_size={args.batch_size}/cache_len={args.cache_len}/"
        f"q_len_per_request={args.q_len_per_request}/num_q_heads={args.num_q_heads}/"
        f"qk_nope_head_dim={args.qk_nope_head_dim}/qk_rope_head_dim={args.qk_rope_head_dim}/"
        f"kv_lora_rank={args.kv_lora_rank}/engine={args.engine}/"
    )

    b = Bench(
        name,
        iters=1,
        met=met_sec,
        metric_list=[bytes_per_sec],
    )

    b.dump_report(output_path=args.output)
