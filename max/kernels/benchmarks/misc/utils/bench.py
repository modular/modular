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

# Ported from deepgemm https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
# Prefer using bench_kineto as it gets profile form a trace. Deepgemm's gmm
# creates a transpose inside their python API. Kineto can exclude the transpose
# and extract the matmul time.

import os
import sys
from collections.abc import Callable
from typing import Any

import torch


def setup_ninja_path() -> None:
    """Add ninja binary to PATH for FlashInfer JIT compilation.

    FlashInfer uses ninja for JIT compilation. In Bazel's pycross_wheel_library
    environment, ninja.BIN_DIR is empty, so we find the binary relative to the
    package location. This must be called before importing flashinfer.
    """
    try:
        import ninja  # type: ignore[import-not-found]

        ninja_bin_dir = ninja.BIN_DIR
        if not ninja_bin_dir:
            # In Bazel pycross_wheel_library, bin is at ../../bin relative to package
            ninja_bin_dir = os.path.normpath(
                os.path.join(os.path.dirname(ninja.__file__), "..", "..", "bin")
            )
        if ninja_bin_dir and os.path.isdir(ninja_bin_dir):
            if ninja_bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = (
                    ninja_bin_dir + ":" + os.environ.get("PATH", "")
                )
    except ImportError:
        pass  # ninja not available, flashinfer import will fail separately


def bench(
    fn: Callable[[], Any],
    num_warmups: int = 5,
    num_tests: int = 10,
    high_precision: bool = False,
) -> float:
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
        y = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
        x @ y

    # Testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3


class empty_suppress:
    def __enter__(self) -> "empty_suppress":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class suppress_stdout_stderr:
    def __enter__(self) -> "suppress_stdout_stderr":
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_: Any) -> None:
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn: Callable[[], Any],
    kernel_names: str | tuple[str, ...],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str | None = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
) -> float | tuple[float, ...]:
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get("DG_NSYS_PROFILING", 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(1e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    assert not isinstance(profiler, empty_suppress)
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # At this point, profiler must be torch.profiler.profile (not empty_suppress)
    assert not isinstance(profiler, empty_suppress)

    # Parse the profiling table
    assert isinstance(kernel_names, str | tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_table = profiler.key_averages().table(
        sort_by="cuda_time_total", max_name_column_width=100
    )
    prof_lines = prof_table.split("\n")

    # Print kineto output to stdout if not suppressed
    if not suppress_kineto_output:
        print(prof_table)

    kernel_names = (
        (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    )
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, (
                f"Errors of the kernel {name} in the profiling table"
            )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                parts = line.split()
                if len(parts) < 2:
                    continue  # Skip malformed lines
                time_str = parts[-2]
                num_str = parts[-1]
                # Debug: print what we're parsing
                # print(f"DEBUG: name={name}, time_str={time_str}, num_str={num_str}")
                for unit, scale in units.items():
                    if unit in time_str:
                        try:
                            total_time += (
                                float(time_str.replace(unit, ""))
                                / scale
                                * int(num_str)
                            )
                            total_num += int(num_str)
                        except ValueError:
                            pass  # Skip lines that don't parse correctly
                        break
        if total_num == 0:
            raise RuntimeError(
                f"No kernel times found for '{name}'. "
                f"Profiler table:\n{prof_table}"
            )
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


# Global flag for CUPTI warmup state
_cupti_warmup_done = False


def bench_kineto_with_cupti_warmup(
    fn: Callable[[], Any],
    kernel_names: str | tuple[str, ...],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str | None = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
) -> float | tuple[float, ...]:
    """Wrapper around bench_kineto that handles CUPTI warmup for CUTLASS kernels.

    On Hopper/Blackwell (SM90+), CUTLASS kernels are launched via cudaLaunchKernelExC
    which CUPTI doesn't capture on the first profiling call due to lazy initialization.
    This wrapper runs a dummy profiling pass first (discarding results) so subsequent
    calls are properly captured.

    Args:
        fn: Function to benchmark
        kernel_names: Name(s) of kernel(s) to measure
        num_tests: Number of test iterations
        suppress_kineto_output: Whether to suppress profiler output
        trace_path: Optional path to save chrome trace
        flush_l2: Whether to flush L2 cache between iterations
        with_multiple_kernels: Whether multiple kernels with the same name are expected

    Returns:
        Average kernel time in seconds (or tuple of times if kernel_names is a tuple)
    """
    global _cupti_warmup_done

    # Run a warmup profiling pass on the first call
    if not _cupti_warmup_done:
        try:
            bench_kineto(
                fn,
                kernel_names=kernel_names,
                num_tests=1,
                suppress_kineto_output=True,
                flush_l2=flush_l2,
                with_multiple_kernels=True,  # Don't assert on first pass
            )
        except (AssertionError, ZeroDivisionError):
            # Expected: first call may not capture kernel or may have zero count
            pass
        _cupti_warmup_done = True

    # Now run the actual benchmark
    return bench_kineto(
        fn,
        kernel_names=kernel_names,
        num_tests=num_tests,
        suppress_kineto_output=suppress_kineto_output,
        trace_path=trace_path,
        flush_l2=flush_l2,
        with_multiple_kernels=with_multiple_kernels,
    )
