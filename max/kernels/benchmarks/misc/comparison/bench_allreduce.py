#!/usr/bin/env python3
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

"""Benchmark comparing NCCL vs SGLang custom allreduce.

This benchmark compares:
- NCCL's allreduce (via torch.distributed)
- SGLang's custom_all_reduce (cross_device_reduce_1stage/2stage)

SGLang's custom allreduce is optimized for small message sizes (<256KB) and uses
P2P communication with NVLink for low latency.

For MAX allreduce benchmarks, use kbench:
    kbench bench_allreduce.yaml --param num_bytes:[16384,262144,1048576]

Usage:
    # Set up virtual environment
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install torch  # or your preferred PyTorch version with CUDA support
    pip install "sglang[all]"  # SGLang with all dependencies

    # Run with torchrun for multi-GPU
    torchrun --nproc_per_node=4 bench_allreduce.py --num-bytes 16384 65536 262144
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist
from max.support.human_readable_formatter import to_human_readable_bytes

# Try importing SGLang's custom allreduce
_sgl_kernel_available = False
_CustomAllreduceClass: type | None = None

# First try the high-level wrapper from sglang package (preferred)
try:
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (  # type: ignore[import-not-found]
        CustomAllreduce,
    )

    _CustomAllreduceClass = CustomAllreduce
except ImportError:
    pass

# Fall back to low-level sgl_kernel API if high-level not available
if _CustomAllreduceClass is None:
    import importlib.util

    _sgl_kernel_available = importlib.util.find_spec("sgl_kernel") is not None


def _bench_cuda_events(
    fn: Any,
    num_warmups: int = 10,
    num_iters: int = 100,
    reduce_max: bool = True,
) -> float:
    """Simple CUDA event-based benchmarking for distributed scenarios.

    Args:
        fn: Function to benchmark
        num_warmups: Number of warmup iterations
        num_iters: Number of test iterations
        reduce_max: If True and distributed is initialized, gather max time across all ranks.
                   For collective operations like allreduce, the operation is only complete
                   when the slowest GPU finishes, so we report max time across all ranks.

    Returns:
        Average time in seconds (max across all ranks if reduce_max=True)
    """
    # Synchronize all processes before warmup
    if dist.is_initialized():
        dist.barrier()

    # Warmup
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    # Synchronize all processes before timing
    if dist.is_initialized():
        dist.barrier()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    # Get local average time in seconds
    local_time_s = start_event.elapsed_time(end_event) / num_iters / 1000.0

    # For collective operations, report the max time across all ranks
    # This is important because allreduce is only complete when ALL GPUs finish
    if reduce_max and dist.is_initialized():
        # Use a tensor to gather max time across all ranks
        time_tensor = torch.tensor(
            [local_time_s], dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        return time_tensor.item()

    return local_time_s


_sglang_parallel_state_initialized = False


def _init_sglang_parallel_state() -> None:
    """Initialize SGLang's parallel state if not already done."""
    global _sglang_parallel_state_initialized
    if _sglang_parallel_state_initialized:
        return

    try:
        from sglang.srt.distributed import (  # type: ignore[import-not-found]
            parallel_state,
        )

        # First initialize the distributed environment (creates world group)
        parallel_state.init_distributed_environment()

        # Then initialize model parallel groups with tensor parallelism matching world size
        world_size = dist.get_world_size()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size
        )
        _sglang_parallel_state_initialized = True
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"  [SGLang] Failed to initialize parallel state: {e}")


def bench_sglang_allreduce(
    num_gpus: int,
    num_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float] | None:
    """Benchmark SGLang's custom allreduce.

    Note: SGLang's custom allreduce requires torch.distributed to be initialized
    and works best with NVLink-connected GPUs.

    Args:
        num_gpus: Number of GPUs to use
        num_bytes: Number of bytes to allreduce
        dtype: Data type for the tensors

    Returns:
        Tuple of (time_seconds, bandwidth_gbps) or None if not available
    """
    if not _sgl_kernel_available and _CustomAllreduceClass is None:
        return None

    # SGLang's custom allreduce requires distributed initialization
    if not dist.is_initialized():
        return None

    # Initialize SGLang's parallel state (required for CustomAllreduce)
    _init_sglang_parallel_state()

    bytes_per_element = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    num_elements = num_bytes // bytes_per_element

    # Create input tensor
    input_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    try:
        if _CustomAllreduceClass is not None:
            # Use high-level CustomAllreduce wrapper from sglang package
            rank = dist.get_rank()

            # Create a CPU group for coordination (CustomAllreduce needs this)
            try:
                cpu_group = dist.new_group(backend="gloo")
            except Exception:
                cpu_group = dist.group.WORLD

            device = torch.device(f"cuda:{rank}")

            custom_ar = _CustomAllreduceClass(
                cpu_group, device, max_size=num_bytes * 2
            )

            if custom_ar.disabled:
                return None

            def run_kernel() -> torch.Tensor:
                return custom_ar.custom_all_reduce(input_tensor)

        elif _sgl_kernel_available:
            # Low-level sgl_kernel API requires multi-process IPC setup
            # which is complex - recommend using sglang[all] instead
            return None

        # Benchmark using CUDA events
        time_s = _bench_cuda_events(run_kernel, num_warmups=10, num_iters=100)

        # Calculate bandwidth
        # AllReduce: each GPU sends and receives (N-1)/N of the data
        # Bus bandwidth = 2 * data_size * (N-1) / N / time
        algbw = num_bytes / time_s / 1e9  # GB/s
        busbw = 2 * algbw * (num_gpus - 1) / num_gpus

        return time_s, busbw

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"SGLang benchmark failed: {e}")
        return None


def bench_nccl_allreduce(
    num_gpus: int,
    num_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float] | None:
    """Benchmark NCCL allreduce as baseline.

    Args:
        num_gpus: Number of GPUs to use
        num_bytes: Number of bytes to allreduce
        dtype: Data type for the tensors

    Returns:
        Tuple of (time_seconds, bandwidth_gbps) or None if not available
    """
    if not dist.is_initialized():
        return None

    bytes_per_element = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    num_elements = num_bytes // bytes_per_element

    input_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    def run_kernel() -> None:
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)

    try:
        time_s = _bench_cuda_events(run_kernel, num_warmups=10, num_iters=100)

        algbw = num_bytes / time_s / 1e9
        busbw = 2 * algbw * (num_gpus - 1) / num_gpus

        return time_s, busbw

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"NCCL benchmark failed: {e}")
        return None


def run_comparison(
    num_gpus: int,
    num_bytes_list: list[int],
    dtype: torch.dtype = torch.bfloat16,
    skip_sglang: bool = False,
    skip_nccl: bool = False,
) -> None:
    """Run allreduce comparison benchmark.

    All ranks must call this function for distributed benchmarks to work.
    Only rank 0 prints results.

    Args:
        num_gpus: Number of GPUs
        num_bytes_list: List of buffer sizes to test
        dtype: Data type
        skip_sglang: Skip SGLang benchmark
        skip_nccl: Skip NCCL benchmark
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    # Collect results for CSV output
    results: list[dict[str, Any]] = []

    if is_main:
        print("=" * 80)
        print(
            f"AllReduce Benchmark: NCCL vs SGLang (num_gpus={num_gpus}, dtype={dtype})"
        )
        print("=" * 80)
        print()

        # Check availability
        sglang_available = (
            _CustomAllreduceClass is not None or _sgl_kernel_available
        )
        dist_available = dist.is_initialized()

        print("Implementation availability:")
        print(
            f"  SGLang custom allreduce:  {'Yes' if sglang_available else 'No'}"
        )
        print(
            f"  torch.distributed (NCCL): {'Yes' if dist_available else 'No'}"
        )
        print()

        if not sglang_available:
            print("To install SGLang kernel:")
            print("  pip install sglang[all]")
            print()

        if not dist_available:
            print("To run NCCL/SGLang benchmarks:")
            print(
                f"  torchrun --nproc_per_node={num_gpus} bench_allreduce.py [args]"
            )
            print()

        # Print header
        print(
            f"{'Size':<10} {'NCCL (ms)':<12} {'NCCL BW':<14} "
            f"{'SGLang (ms)':<12} {'SGLang BW':<14} {'SGLang/NCCL':<12}"
        )
        print("-" * 80)

    for num_bytes in num_bytes_list:
        # All ranks must participate in distributed benchmarks
        nccl_result = (
            None
            if skip_nccl
            else bench_nccl_allreduce(num_gpus, num_bytes, dtype)
        )
        sglang_result = (
            None
            if skip_sglang
            else bench_sglang_allreduce(num_gpus, num_bytes, dtype)
        )

        # Only rank 0 prints results
        if is_main:
            size_str = to_human_readable_bytes(num_bytes)

            nccl_time_str = "N/A"
            nccl_bw_str = "N/A"
            sglang_time_str = "N/A"
            sglang_bw_str = "N/A"
            speedup_str = "N/A"

            # Store raw values for CSV
            row: dict[str, Any] = {
                "num_bytes": num_bytes,
                "size": size_str,
                "nccl_time_ms": None,
                "nccl_bw_gbps": None,
                "sglang_time_ms": None,
                "sglang_bw_gbps": None,
                "speedup": None,
            }

            if nccl_result:
                nccl_time, nccl_bw = nccl_result
                nccl_time_str = f"{nccl_time * 1000:.4f}"
                nccl_bw_str = f"{nccl_bw:.2f} GB/s"
                row["nccl_time_ms"] = nccl_time * 1000
                row["nccl_bw_gbps"] = nccl_bw

            if sglang_result:
                sglang_time, sglang_bw = sglang_result
                sglang_time_str = f"{sglang_time * 1000:.4f}"
                sglang_bw_str = f"{sglang_bw:.2f} GB/s"
                row["sglang_time_ms"] = sglang_time * 1000
                row["sglang_bw_gbps"] = sglang_bw

            if nccl_result and sglang_result:
                speedup = nccl_result[0] / sglang_result[0]
                speedup_str = f"{speedup:.2f}x"
                row["speedup"] = speedup

            results.append(row)

            print(
                f"{size_str:<10} {nccl_time_str:<12} {nccl_bw_str:<14} "
                f"{sglang_time_str:<12} {sglang_bw_str:<14} {speedup_str:<12}"
            )

    if is_main:
        print("=" * 80)
        print()
        print("Notes:")
        print(
            "  - NCCL: NVIDIA Collective Communication Library (via torch.distributed)"
        )
        print(
            "  - SGLang: Custom allreduce using NVLink P2P (optimized for <256KB)"
        )
        print("  - Bus bandwidth = 2 * algbw * (N-1) / N")
        print(
            "  - Times are max across all GPUs (collective is complete when slowest finishes)"
        )
        print()
        print("  For MAX allreduce benchmarks, use kbench:")
        print(
            "    kbench bench_allreduce.yaml --param num_bytes:[16384,262144]"
        )

        # Print CSV output
        print()
        print("=" * 80)
        print("CSV OUTPUT")
        print("=" * 80)
        print(
            "num_bytes,size,nccl_time_ms,nccl_bw_gbps,sglang_time_ms,sglang_bw_gbps,speedup"
        )
        for row in results:
            nccl_t = (
                f"{row['nccl_time_ms']:.4f}"
                if row["nccl_time_ms"] is not None
                else ""
            )
            nccl_b = (
                f"{row['nccl_bw_gbps']:.2f}"
                if row["nccl_bw_gbps"] is not None
                else ""
            )
            sgl_t = (
                f"{row['sglang_time_ms']:.4f}"
                if row["sglang_time_ms"] is not None
                else ""
            )
            sgl_b = (
                f"{row['sglang_bw_gbps']:.2f}"
                if row["sglang_bw_gbps"] is not None
                else ""
            )
            spd = f"{row['speedup']:.2f}" if row["speedup"] is not None else ""
            print(
                f"{row['num_bytes']},{row['size']},{nccl_t},{nccl_b},{sgl_t},{sgl_b},{spd}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AllReduce benchmark: NCCL vs SGLang"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: auto-detect from WORLD_SIZE)",
    )
    parser.add_argument(
        "--num-bytes",
        type=int,
        nargs="+",
        default=[
            16 * 1024,
            64 * 1024,
            256 * 1024,
            1024 * 1024,
            16 * 1024 * 1024,
        ],
        help="Buffer sizes in bytes to test",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--skip-sglang",
        action="store_true",
        help="Skip SGLang benchmark",
    )
    parser.add_argument(
        "--skip-nccl",
        action="store_true",
        help="Skip NCCL benchmark",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Initialize distributed if running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )

    # Determine number of GPUs
    if args.num_gpus:
        num_gpus = args.num_gpus
    elif dist.is_initialized():
        num_gpus = dist.get_world_size()
    else:
        num_gpus = torch.cuda.device_count()

    if not dist.is_initialized():
        print(
            "Error: This benchmark requires torchrun for multi-GPU execution."
        )
        print(
            f"  torchrun --nproc_per_node={num_gpus} bench_allreduce.py [args]"
        )
        return

    # All ranks must call run_comparison for distributed benchmarks
    # (only rank 0 prints results)
    run_comparison(
        num_gpus=num_gpus,
        num_bytes_list=args.num_bytes,
        dtype=dtype_map[args.dtype],
        skip_sglang=args.skip_sglang,
        skip_nccl=args.skip_nccl,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
