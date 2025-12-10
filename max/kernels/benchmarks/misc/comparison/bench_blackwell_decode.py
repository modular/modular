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

# Setup: run setup_bench_env.py (MAX installs by default) then activate the
# produced venv. Example:
#   python $MODULAR_PATH/Kernels/benchmarks/comparison/setup_bench_env.py
#   source $MODULAR_PATH/.venv/bin/activate
# The only SoTA MHA decode kernel on blackwell is from TRTLLM, called via flashinfer.
# Run via Bazel: br //max/kernels/benchmarks/misc/comparison:bench_decode

from __future__ import annotations

import math
import os
import sys
import types
from typing import Any

import torch

# Import bench utilities from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import bench_kineto, setup_ninja_path

# MAX imports
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedCacheValues

# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None


def bench_flashinfer(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    use_tensor_cores: bool = True,
    backend: str = "fa",
) -> tuple[float, float] | None:
    """Benchmark FlashInfer decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        block_size: Page/block size for paged KV cache
        dtype: torch dtype for inputs
        use_tensor_cores: Whether to use tensor cores
        backend: Backend to use ("fa" for FlashAttention, "trtllm" for TensorRT-LLM, "trtllm-gen" for TensorRT-LLM generation)
    """
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    # Determine layout based on backend
    kv_layout = "HND" if backend in ["trtllm", "trtllm-gen"] else "NHD"

    # Query tensor (one token per sequence for decode)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # Create block tables
    max_num_blocks_per_seq = (cache_len + block_size - 1) // block_size
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int, device="cuda"
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Build FlashInfer metadata
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []

    for i in range(batch_size):
        num_blocks = (cache_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)

        kv_last_page_len = cache_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.tensor(
        kv_last_page_lens, dtype=torch.int32, device="cuda"
    )

    # Create paged KV cache with layout matching the backend
    if kv_layout == "HND":
        # TensorRT-LLM backend expects [num_blocks, 2, num_kv_heads, block_size, head_dim]
        key_value_cache = torch.randn(
            batch_size * max_num_blocks_per_seq,
            2,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
    else:
        # FlashAttention backend expects [num_blocks, 2, block_size, num_kv_heads, head_dim]
        key_value_cache = torch.randn(
            batch_size * max_num_blocks_per_seq,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )

    # Create decode wrapper
    wrapper = _flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=use_tensor_cores,
        backend=backend,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        "NONE",  # position encoding mode
        q_data_type=dtype,
    )

    def run_kernel() -> torch.Tensor:
        return wrapper.forward(q, key_value_cache)

    # Warmup
    for _ in range(10):
        wrapper.forward(q, key_value_cache)

    # Now benchmark with the correct kernel name
    # TensorRT-LLM backend uses BatchPrefillWithPagedKVCacheKernel (even for decode)
    # FlashAttention backend uses BatchDecodeWithPagedKVCacheKernel
    if backend in ["trtllm", "trtllm-gen"]:
        kernel_name = "BatchPrefillWithPagedKVCacheKernel"
    else:
        kernel_name = "BatchDecodeWithPagedKVCacheKernel"

    time_s = bench_kineto(
        run_kernel,
        kernel_names=kernel_name,
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    # Calculate memory throughput (GB/s)
    # Read: q (batch_size * num_heads * head_dim) + k,v (batch_size * cache_len * num_kv_heads * head_dim * 2)
    # Write: output (batch_size * num_heads * head_dim)
    bytes_per_element = (
        2 if dtype == torch.bfloat16 or dtype == torch.float16 else 4
    )
    gb_per_sec = (
        2
        * batch_size
        * (num_heads * head_dim + cache_len * num_kv_heads * head_dim)
        * bytes_per_element
        / time_s
        / 1e9
    )

    return time_s, gb_per_sec


def bench_max(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Benchmark MAX flash_attention_ragged with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        page_size: Page size for paged KV cache
        dtype: torch dtype for inputs
    """
    # Convert torch dtype to MAX DType
    max_dtype = DType.from_torch(dtype)

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # Setup KV cache configuration
    kv_params = KVCacheParams(
        dtype=max_dtype,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
        n_devices=1,
    )

    # Calculate required memory:
    # batch_size * num_blocks_per_seq * block_size * num_kv_heads * num_layers * head_dim * dtype_size
    num_blocks_per_seq = (cache_len + page_size - 1) // page_size
    bytes_per_element = (
        2 if max_dtype == DType.bfloat16 or max_dtype == DType.float16 else 4
    )
    # 2x for K and V caches
    required_memory = (
        2
        * batch_size
        * num_blocks_per_seq
        * page_size
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )

    # [num_pages, 2 for K and V, 1 layer, ...]
    paged_blocks_torch = torch.randn(
        batch_size * num_blocks_per_seq,
        2,
        1,
        page_size,
        num_kv_heads,
        head_dim,
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
    # max_lengths shape: [num_steps, 2] where column 0 is max_seq_length, column 1 is max_cache_length
    # For decode: max_seq_length=1 (one token), max_cache_length=cache_len
    max_lengths_torch = torch.tensor(
        [[1, cache_len]], dtype=torch.uint32, device="cpu"
    )

    # Convert torch tensors to MAX types (these will be the actual runtime inputs)
    paged_blocks_max = Tensor.from_dlpack(
        paged_blocks_torch
    )  # Buffer for kv_blocks
    lut_max = Tensor.from_dlpack(lut_torch)  # Tensor for lookup_table
    cache_lengths_max = Tensor.from_dlpack(
        cache_lengths_torch
    )  # Tensor for cache_lengths
    max_lengths_max = Tensor.from_dlpack(
        max_lengths_torch
    )  # Tensor for max_lengths

    # Define input types.
    # Avoid using KVCacheManager for its complecity.
    # For decode: query is [total_tokens, num_heads, head_dim] where total_tokens = batch_size
    q_type = TensorType(
        max_dtype,
        shape=["total_tokens", num_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["batch_size_plus_1"],
        device=DeviceRef.GPU(),
    )

    blocks_type = BufferType(
        max_dtype,
        shape=["total_num_pages", 2, 1, page_size, num_kv_heads, head_dim],
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

    # Build graph with paged KV cache inputs
    with Graph(
        "flash_attn_decode_max",
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

        result = flash_attention_ragged(
            kv_params,
            q.tensor,
            input_row_offsets.tensor,
            kv_collection,
            layer_idx,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / math.sqrt(head_dim),
        )

        graph.output(result)

    # Compile model
    model = session.load(graph)

    # Prepare inputs - for decode, each sequence has 1 token
    # Total tokens = batch_size
    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Input row offsets: [0, 1, 2, ..., batch_size] (one token per sequence)
    input_row_offsets = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
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

    # Use bench_kineto to profile the kernel
    time_s = bench_kineto(
        run_kernel,
        kernel_names="mha",
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    gb_per_sec = (
        2
        * batch_size
        * (num_heads * head_dim + cache_len * num_kv_heads * head_dim)
        * 2
        / time_s
        / 1e9
    )

    return time_s, gb_per_sec


def bench_decode(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    """Run all MHA decode benchmarks and display results side-by-side.

    Args:
        batch_size: Batch size (number of sequences)
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
    """
    print("=" * 80)
    print(
        f"MHA Decode Benchmark (batch={batch_size}, cache_len={cache_len}, "
        f"q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim})"
    )
    print("=" * 80)

    results: dict[str, tuple[float, float] | None] = {}

    # Run FlashInfer benchmark with TensorRT-LLM backend
    if _flashinfer is not None:
        try:
            result = bench_flashinfer(
                batch_size,
                cache_len,
                num_heads,
                num_kv_heads,
                head_dim,
                64,  # TRTLLM's page table size
                dtype,
                use_tensor_cores=True,
                backend="trtllm",
            )
            results["flashinfer"] = result
        except Exception as e:
            print(f"FlashInfer benchmark failed: {e}")
            import traceback

            traceback.print_exc()
            results["flashinfer"] = None
    else:
        results["flashinfer"] = None

    # Run MAX benchmark
    try:
        result = bench_max(
            batch_size,
            cache_len,
            num_heads,
            num_kv_heads,
            head_dim,
            128,  # MAX's page size
            dtype,
        )
        results["max"] = result
    except Exception as e:
        print(f"MAX benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        results["max"] = None

    # Print results
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'GB/s':<15}")
    print("-" * 50)

    # FlashInfer
    if results["flashinfer"] is not None:
        time_s, gb_per_sec = results["flashinfer"]
        print(f"{'FlashInfer':<20} {time_s * 1000:<15.4f} {gb_per_sec:<15.2f}")
    else:
        print(f"{'FlashInfer':<20} {'N/A':<15} {'N/A':<15}")

    # MAX
    if results["max"] is not None:
        time_s, gb_per_sec = results["max"]
        print(f"{'MAX':<20} {time_s * 1000:<15.4f} {gb_per_sec:<15.2f}")
    else:
        print(f"{'MAX':<20} {'N/A':<15} {'N/A':<15}")

    print("=" * 80)


if __name__ == "__main__":
    # Decode benchmark: batch_size, cache_len, num_heads, num_kv_heads, head_dim, page_size
    bench_decode(
        batch_size=128,
        cache_len=1024,
        num_heads=4,
        num_kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
    )
