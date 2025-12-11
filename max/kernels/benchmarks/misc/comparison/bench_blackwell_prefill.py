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

# Blackwell prefill benchmark comparing MAX against FlashInfer and flash-attention baselines.
# Run via Bazel: br //max/kernels/benchmarks/misc/comparison:bench_prefill

from __future__ import annotations

import math
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

# Import bench utilities from Bazel dependency (bench_utils target)
from bench import bench_kineto_with_cupti_warmup, setup_ninja_path

# MAX imports
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu

# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None

_flash_attn_varlen_func: Callable[..., Any] | None
try:
    # The pure Python flash-attention wheel's __init__.py tries to import
    # flash_attn_2_cuda (CUDA extension not in pure wheel).
    # Bypass this by creating a stub flash_attn module with valid __path__
    # but no imports.
    import importlib.util
    import sys

    flash_attn_spec = importlib.util.find_spec("flash_attn")
    if (
        flash_attn_spec is None
        or flash_attn_spec.submodule_search_locations is None
    ):
        raise ImportError("flash_attn package not found")

    # Create stub module with valid __path__ but no imports
    flash_attn_stub = types.ModuleType("flash_attn")
    flash_attn_stub.__path__ = list(flash_attn_spec.submodule_search_locations)
    flash_attn_stub.__file__ = flash_attn_spec.origin
    sys.modules["flash_attn"] = flash_attn_stub

    # Now import the cute subpackage and interface
    from flash_attn.cute.interface import (
        flash_attn_varlen_func as _flash_attn_varlen_func,
    )
except ImportError as e:
    print(f"Error: flash_attn not available: {e}")
    _flash_attn_varlen_func = None


def bench_flashinfer(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    backend: str = "cutlass",
) -> tuple[float, float] | None:
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    # Validate backend option
    available_backends = ["auto", "fa2", "fa3", "trtllm-gen", "cutlass"]
    assert backend in available_backends, (
        f"backend must be one of {available_backends}, got {backend}"
    )

    # Note: trtllm-gen backend doesn't support variable length yet
    if backend == "trtllm-gen":
        print(
            "Warning: trtllm-gen backend doesn't support variable length yet, skipping..."
        )
        return None

    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * qkv_len
    )
    kv_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * qkv_len
    )
    wrapper = _flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(
            128 * 1024 * 1024, dtype=dtype, device="cuda"
        ),  # work space
        kv_layout="NHD",
        backend=backend,
    )
    wrapper.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        num_heads,
        num_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    def run_kernel() -> torch.Tensor:
        return wrapper.run(q, k, v)

    # Select kernel name pattern based on backend
    if backend == "cutlass":
        kernel_name = "Sm100FmhaFwdKernelTmaWarpspecialized"
    elif backend == "fa3":
        kernel_name = "fmha"  # FlashAttention 3 kernel pattern
    elif backend == "fa2":
        kernel_name = "fmha"  # FlashAttention 2 kernel pattern
    else:  # auto
        kernel_name = "fmha"  # Generic pattern

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup for CUTLASS
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names=kernel_name,
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops(time_s: float) -> float:
        if causal:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 2
                / time_s
                / 1e12
            )
        else:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 4
                / time_s
                / 1e12
            )

    # print(
    #     f"bench_flashinfer (batch_size={batch_size}, qkv_len={qkv_len}, num_heads={num_heads}, head_dim={head_dim}, causal={causal}), time={time_s*1000:.4f} ms, flops: {flops(time_s):.4f} TFLOPs/s"
    # )

    return time_s, flops(time_s)


def bench_max(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> tuple[float, float] | None:
    """Benchmark MAX flash_attention_gpu kernel.

    Args:
        batch_size: Batch size
        qkv_len: Sequence length for Q, K, V
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
    """
    # Convert torch dtype to MAX DType
    max_dtype = DType.from_torch(dtype)

    # Create input tensors in (batch, seq_len, num_heads, head_dim) format
    q = torch.randn(
        batch_size, qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Define tensor types for MAX graph
    q_type = TensorType(
        max_dtype,
        shape=["batch", "seq_len", num_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        max_dtype,
        shape=["batch", "seq_len", num_heads, head_dim],
        device=DeviceRef.GPU(),
    )

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # Construct MAX graph
    mask_variant = (
        MHAMaskVariant.CAUSAL_MASK if causal else MHAMaskVariant.NULL_MASK
    )
    graph = Graph(
        "flash_attn_max",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=mask_variant,
        ),
        input_types=[q_type, kv_type, kv_type],
    )

    # Compile model
    model = session.load(graph)

    def run_kernel() -> torch.Tensor:
        output = model.execute(q.detach(), k.detach(), v.detach())[0]
        return output

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="mha",
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops(time_s: float) -> float:
        if causal:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 2
                / time_s
                / 1e12
            )
        else:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 4
                / time_s
                / 1e12
            )

    return time_s, flops(time_s)


def bench_tridao(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> tuple[float, float] | None:
    if _flash_attn_varlen_func is None:
        print("flash_attn not available, skipping bench_tridao")
        return None

    # Create input tensors in varlen format (similar to test_flash_attn_varlen_output)
    q = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * qkv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Create cumulative sequence length offsets
    cu_seqlens_q = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * qkv_len
    )
    cu_seqlens_k = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * qkv_len
    )

    def run_kernel() -> torch.Tensor:
        assert _flash_attn_varlen_func is not None
        out, _ = _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=causal,
            pack_gqa=False,
        )
        return out

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100",
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops(time_s: float) -> float:
        if causal:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 2
                / time_s
                / 1e12
            )
        else:
            return (
                batch_size
                * qkv_len
                * qkv_len
                * num_heads
                * head_dim
                * 4
                / time_s
                / 1e12
            )

    return time_s, flops(time_s)


def bench_prefill(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    """Run all MHA prefill benchmarks and display results side-by-side.

    Args:
        batch_size: Batch size
        qkv_len: Sequence length for Q, K, V
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
    """
    print("=" * 80)
    print(
        f"MHA Prefill Benchmark (batch={batch_size}, seq_len={qkv_len}, heads={num_heads}, head_dim={head_dim}, causal={causal})"
    )
    print("=" * 80)

    results: dict[str, tuple[float, float] | None] = {}

    # Run FlashInfer benchmark
    if _flashinfer is not None:
        try:
            results["flashinfer"] = bench_flashinfer(
                batch_size,
                qkv_len,
                num_heads,
                head_dim,
                causal,
                dtype,
                backend="cutlass",
            )
        except Exception as e:
            print(f"FlashInfer benchmark failed: {e}")
            results["flashinfer"] = None
    else:
        results["flashinfer"] = None

    # Run Tri Dao benchmark
    if _flash_attn_varlen_func is not None:
        try:
            results["tridao"] = bench_tridao(
                batch_size, qkv_len, num_heads, head_dim, causal, dtype
            )
        except Exception as e:
            print(f"Tri Dao benchmark failed: {e}")
            results["tridao"] = None
    else:
        results["tridao"] = None

    print(f"{'Implementation':<20} {'Time (ms)':<15} {'TFLOPs/s':<15}")

    # Run MAX benchmark
    try:
        results["max"] = bench_max(
            batch_size, qkv_len, num_heads, head_dim, causal, dtype
        )
    except Exception as e:
        print(f"MAX benchmark failed: {e}")
        results["max"] = None

    # FlashInfer
    if results["flashinfer"] is not None:
        time_s, tflops = results["flashinfer"]
        print(f"{'FlashInfer':<20} {time_s * 1000:<15.4f} {tflops:<15.2f}")
    else:
        print(f"{'FlashInfer':<20} {'N/A':<15} {'N/A':<15}")

    # Tri Dao
    if results["tridao"] is not None:
        time_s, tflops = results["tridao"]
        print(
            f"{'Tri Dao Flash-Attn':<20} {time_s * 1000:<15.4f} {tflops:<15.2f}"
        )
    else:
        print(f"{'Tri Dao Flash-Attn':<20} {'N/A':<15} {'N/A':<15}")

    # MAX
    if results["max"] is not None:
        time_s, tflops = results["max"]
        print(f"{'MAX':<20} {time_s * 1000:<15.4f} {tflops:<15.2f}")
    else:
        print(f"{'MAX':<20} {'N/A':<15} {'N/A':<15}")

    print("=" * 80)


if __name__ == "__main__":
    bench_prefill(1, 4096, 32, 128, False, torch.bfloat16)
    bench_prefill(2, 4096, 32, 128, False, torch.bfloat16)
    bench_prefill(1, 8192, 32, 128, False, torch.bfloat16)
