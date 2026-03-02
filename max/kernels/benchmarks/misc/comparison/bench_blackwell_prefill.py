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

# Blackwell prefill benchmark comparing MAX against FlashInfer and flash-attention baselines.
# Run via kbench: kbench bench_prefill.yaml

from __future__ import annotations

import argparse
import math
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

# Import bench utilities from Bazel dependency (bench_utils target)
from bench import bench_kineto_with_cupti_warmup, setup_ninja_path
from bencher_utils import Bench, ThroughputMeasure

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
    num_iters: int,
    backend: str = "cutlass",
    no_kineto: bool = False,
) -> tuple[float, int] | None:
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
            "Warning: trtllm-gen backend doesn't support variable length yet,"
            " skipping..."
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

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup for CUTLASS
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names=kernel_name,
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops() -> int:
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4

    return time_s, flops()


def bench_max(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
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

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="mha",
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops() -> int:
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4

    return time_s, flops()


def bench_tridao(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
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

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100",
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    def flops() -> int:
        if causal:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 2
        else:
            return batch_size * qkv_len * qkv_len * num_heads * head_dim * 4

    return time_s, flops()


def bench_prefill(
    batch_size: int,
    qkv_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    engine: str,
    num_iters: int,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    """Run all MHA prefill benchmarks and display results side-by-side.

    Args:
        batch_size: Batch size
        qkv_len: Sequence length for Q, K, V
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
        engine: backend to run the benchmark ("flashinfer" or "tridao" or "modular_max")
        num_iters: Number of benchmark iters.
    """
    print("=" * 80)
    print(
        f"MHA Prefill Benchmark (batch={batch_size}, seq_len={qkv_len},"
        f" heads={num_heads}, head_dim={head_dim}, causal={causal})"
    )
    print("=" * 80)

    result: tuple[float, int] | None = None

    if engine == "flashinfer":
        # Run FlashInfer benchmark
        if _flashinfer is not None:
            try:
                result = bench_flashinfer(
                    batch_size,
                    qkv_len,
                    num_heads,
                    head_dim,
                    causal,
                    dtype,
                    num_iters,
                    backend="cutlass",
                    no_kineto=no_kineto,
                )
            except Exception as e:
                print(f"FlashInfer benchmark failed: {e}")

    elif engine == "tridao":
        # Run Tri Dao benchmark
        if _flash_attn_varlen_func is not None:
            try:
                result = bench_tridao(
                    batch_size,
                    qkv_len,
                    num_heads,
                    head_dim,
                    causal,
                    dtype,
                    num_iters,
                    no_kineto,
                )
            except Exception as e:
                print(f"Tri Dao benchmark failed: {e}")

    # Run MAX benchmark
    elif engine == "modular_max":
        try:
            result = bench_max(
                batch_size,
                qkv_len,
                num_heads,
                head_dim,
                causal,
                dtype,
                num_iters,
                no_kineto,
            )
        except Exception as e:
            print(f"MAX benchmark failed: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHA Prefill Benchmark")
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--qkv_len",
        "--qkv-len",
        type=int,
        default=4096,
        help="QKV length",
    )
    parser.add_argument(
        "--num_heads",
        "--num-heads",
        type=int,
        default=32,
        help="Number of query heads",
    )

    parser.add_argument(
        "--head_dim", "--head-dim", type=int, default=128, help="Head dimension"
    )

    parser.add_argument(
        "--causal",
        type=bool,
        default=False,
        help="Causal",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=["flashinfer", "tridao", "modular_max"],
        help="Engine",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )

    parser.add_argument(
        "--num_iters",
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-kineto",
        action="store_true",
        help="Skip kineto timing (for ncu/nsys).",
    )
    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if args.engine not in ["flashinfer", "tridao", "modular_max"]:
        raise ValueError(f"engine {args.engine} is not supported!")

    result = bench_prefill(
        batch_size=args.batch_size,
        qkv_len=args.qkv_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        causal=args.causal,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
        num_iters=args.num_iters,
        no_kineto=args.no_kineto,
    )

    if args.num_iters > 1 and not args.no_kineto:
        met_sec, flops = result if result else [0, 0]
        flops_per_sec = ThroughputMeasure(Bench.flops, flops)
        name = (
            f"MHA_Prefill/batch_size={args.batch_size}/qkv_len={args.qkv_len}/"
            f"num_heads={args.num_heads}/head_dim={args.head_dim}/"
            f"causal={args.causal}/dtype={dtype_map[args.dtype]}/"
            f"engine={args.engine}/"
        )

        b = Bench(
            name,
            iters=1,
            met=met_sec,
            metric_list=[flops_per_sec],
        )

        b.dump_report(output_path=args.output)
