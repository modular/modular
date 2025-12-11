# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from bencher_utils import Bench, ThroughputMeasure, arg_parse
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flare_mla_decode_ragged
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedCacheValues

LINE = "=" * 80

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
    # time_s: float,
    model_config: Config,
) -> float:
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
    # Read: query + kv_cache
    # Write: output
    bytes_per_element = 2 if dtype in [torch.bfloat16, torch.float16] else 4

    # Input: query [batch_size, q_len_per_request, num_q_heads, kv_lora_rank + qk_rope_head_dim]
    query_bytes = (
        batch_size
        * q_len_per_request
        * num_q_heads
        * (kv_lora_rank + qk_rope_head_dim)
        * bytes_per_element
    )

    # KV cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim] (read only what's needed)
    kv_bytes = (
        batch_size
        * cache_len
        * (kv_lora_rank + qk_rope_head_dim)
        * bytes_per_element
    )

    # Output: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank]
    output_bytes = (
        batch_size
        * q_len_per_request
        * num_q_heads
        * kv_lora_rank
        * bytes_per_element
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
) -> tuple[float, float] | None:
    """Benchmark FlashInfer MLA decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence (max sequence length)
        page_size: Page/block size for paged KV cache
        dtype: torch dtype for inputs
        q_len_per_request: Query length per request (1 for decode, >1 for chunked prefill)
        backend: Backend to use ("trtllm-gen" for TensorRT-LLM generation, "xqa" for XQA)
        enable_pdl: Enable PDL (Persistent Dynamic Load) optimization
    """
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    device = "cuda"

    # DeepSeek MLA configuration
    num_q_heads = model_config.num_q_heads
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_lora_rank = model_config.kv_lora_rank

    # Query tensor: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank + qk_rope_head_dim]
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=dtype,
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
    kv_cache = torch.randn(
        num_blocks,
        page_size,
        kv_lora_rank + qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )

    # Workspace buffer (must be zero-initialized for trtllm-gen backend)
    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    # Scale for attention computation
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

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
            bmm1_scale=scale,
            bmm2_scale=1.0,
            enable_pdl=enable_pdl,
            backend="trtllm-gen",
        )

    # Warmup
    for _ in range(10):
        run_kernel()

    # Benchmark with CUPTI warmup for CUTLASS kernels
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="fmhaSm100",  # FlashInfer TRT-LLM MLA kernel name prefix
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
    )

    return time_s, total_bytes


def bench_max(
    batch_size: int,
    cache_len: int,
    page_size: int,
    dtype: torch.dtype,
    model_config: Config,
    q_len_per_request: int = 1,
) -> tuple[float, float]:
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
        n_devices=1,
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
    paged_blocks_max = Tensor.from_dlpack(paged_blocks_torch)
    lut_max = Tensor.from_dlpack(lut_torch)
    cache_lengths_max = Tensor.from_dlpack(cache_lengths_torch)
    max_lengths_max = Tensor.from_dlpack(max_lengths_torch)

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

    # Benchmark with CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="mla",
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
    )

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
) -> dict:
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

    result: tuple[float, float] | None = None
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

    batch_size = int(arg_parse("batch_size", 128))
    cache_len = int(arg_parse("cache_len", 1024))

    dtype = torch.bfloat16
    q_len_per_request = int(arg_parse("q_len_per_request", 1))

    num_q_heads = int(arg_parse("num_q_heads", cfg.num_q_heads))
    qk_nope_head_dim = int(arg_parse("qk_nope_head_dim", cfg.qk_nope_head_dim))
    qk_rope_head_dim = int(arg_parse("qk_rope_head_dim", cfg.qk_rope_head_dim))
    kv_lora_rank = int(arg_parse("num_q_heads", cfg.kv_lora_rank))

    output_path = arg_parse("output", "output.csv", short_handle="o")

    print("MLA Decode Benchmark (DeepSeek V2/V3)")
    print(
        f"  num_q_heads={cfg.num_q_heads}, qk_nope_dim={cfg.qk_nope_head_dim}, "
        f"qk_rope_dim={cfg.qk_rope_head_dim}, kv_lora_rank={cfg.kv_lora_rank}"
    )
    print("  dtype=torch.bfloat16, q_len=1 (decode)")
    print(LINE)

    # TODO: overlap this with "backend"
    engine = arg_parse("engine", "modular_max")
    if engine not in ["flashinfer", "modular_max"]:
        raise ValueError(f"engine {engine} is not supported!")

    result = bench_mla_decode(
        batch_size=batch_size,
        cache_len=cache_len,
        dtype=dtype,
        engine=engine,
        model_config=cfg,
        q_len_per_request=q_len_per_request,
        backend="trtllm-gen",
        enable_pdl=True,
    )

    met_ms, bytes = result
    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    name = (
        f"MLA/batch_size={batch_size}/cache_len={cache_len}/"
        f"q_len_per_request={q_len_per_request}/num_q_heads={num_q_heads}/"
        f"qk_nope_head_dim={qk_nope_head_dim}/qk_rope_head_dim={qk_rope_head_dim}/"
        f"kv_lora_rank={kv_lora_rank}/engine={engine}/"
    )

    b = Bench(
        name,
        iters=1,
        met=met_ms,
        metric_list=[bytes_per_sec],
    )

    b.dump_report(output_path=output_path)
