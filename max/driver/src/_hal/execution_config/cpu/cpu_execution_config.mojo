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

from std.collections import BitSet
from .. import (
    ExecutionConfig,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    CoresExecutionConfig,
    ParticularCoresExecutionConfig,
    ThreadsPerCoreExecutionConfig,
    ThreadCountExecutionConfig,
    ThreadMaskExecutionConfig,
    OptionalMatrixUnitExecutionConfig,
)
from std.sys.defines import get_defined_int

# Default to the max on Linux for common ISAs at the time of writing.
# TODO: Derive from machine definition
comptime _MAX_CORE_COUNT: Int = get_defined_int[
    "MODULAR_MAX_MAX_CORE_COUNT", 8192
]()

# SMT1 or SMT2 is the vast majority case
comptime _MAX_THREADS_PER_CORE: Int = get_defined_int[
    "MODULAR_MAX_MAX_THREADS_PER_CORE", 2
]()


struct CPUExecutionConfiguration(
    Copyable,
    CoresExecutionConfig,
    ExecutionConfig,
    Movable,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    OptionalMatrixUnitExecutionConfig,
    ParticularCoresExecutionConfig,
    ThreadCountExecutionConfig,
    ThreadMaskExecutionConfig,
    ThreadsPerCoreExecutionConfig,
):
    """Describes the execution configuration of a kernel launch on the CPU.

    The launch is tracked at thread granularity: a single `thread_mask` records
    exactly which hardware threads are enabled, giving finer-grained control
    than a per-core mask. Cores are contiguous groups of `MAX_THREADS_PER_CORE`
    threads, so the core-level views (`get_core_mask`, `get_num_cores`) are
    derived from the thread mask: a core counts as enabled when any of its
    threads are enabled.
    """

    comptime MAX_CORE_COUNT: Int = _MAX_CORE_COUNT
    """The number of cores this device exposes."""

    comptime MAX_THREADS_PER_CORE: Int = _MAX_THREADS_PER_CORE
    """The number of hardware threads per core."""

    comptime MAX_THREAD_COUNT: Int = (
        Self.MAX_CORE_COUNT * Self.MAX_THREADS_PER_CORE
    )
    """The total number of hardware threads across all cores."""

    var thread_mask: BitSet[Self.MAX_THREAD_COUNT]
    var scratchpad_usage: Int
    var enable_matrix_unit: Bool
    """Intended for use with Intel AMX or ACE."""

    def __init__[
        bit_mask_size: Int, //
    ](
        out self,
        var thread_mask: BitSet[bit_mask_size],
        *,
        var scratchpad_usage: Int = 0,
        var enable_matrix_unit: Bool = True,
    ):
        """The 'all members' constructor with reasonable defaults set.

        The thread mask is resized to `MAX_THREAD_COUNT`, dropping any bits set
        at or above `MAX_THREAD_COUNT`.

        Parameters:
            bit_mask_size: The size of the source thread mask.

        Args:
            thread_mask: The mask of threads to enable for the kernel launch.
            scratchpad_usage: The near-compute scratchpad usage in bytes.
            enable_matrix_unit: Whether to use the matrix units.
        """
        self.thread_mask = BitSet[Self.MAX_THREAD_COUNT](
            resized_from=thread_mask
        )
        self.scratchpad_usage = scratchpad_usage
        self.enable_matrix_unit = enable_matrix_unit

    def __init__(out self, *, thread_mask: BitSet[Self.MAX_THREAD_COUNT]):
        """Initializes the execution config with the given thread mask.

        Args:
            thread_mask: The mask of threads to enable for the kernel launch.
        """
        self = Self(thread_mask.copy())

    def __init__(out self, *, num_threads: Int):
        """Initializes the execution config with the given number of threads.

        Enables the first `num_threads` threads.

        Args:
            num_threads: The number of threads to use for the kernel launch.
        """
        var thread_mask = BitSet[Self.MAX_THREAD_COUNT]()
        for i in range(num_threads):
            thread_mask.set(i)
        self = Self(thread_mask^)

    def __init__(out self, *, num_cores: Int):
        """Initializes the execution config with the given number of cores.

        Enables every thread of the first `num_cores` cores.

        `num_cores` is clamped to `MAX_CORE_COUNT`.

        Args:
            num_cores: The number of cores to use for the kernel launch.
        """
        real_num_cores = min(num_cores, Self.MAX_CORE_COUNT)
        self = Self(
            num_cores=real_num_cores,
            num_threads_per_core=Self.MAX_THREADS_PER_CORE,
        )

    def __init__(out self, *, num_cores: Int, num_threads_per_core: Int):
        """Initializes the execution config with the given threads per core.

        Enables the first `num_threads_per_core` threads of each of the first
        `num_cores` cores. `num_threads_per_core` is clamped to
        `MAX_THREADS_PER_CORE`.

        `num_cores` is clamped to `MAX_CORE_COUNT`.

        Args:
            num_cores: The number of cores to use for the kernel launch.
            num_threads_per_core: The number of threads per core to use for the
                kernel launch.
        """
        real_num_cores = min(num_cores, Self.MAX_CORE_COUNT)
        var threads_per_core = min(
            num_threads_per_core, Self.MAX_THREADS_PER_CORE
        )
        var thread_mask = BitSet[Self.MAX_THREAD_COUNT]()
        for core in range(real_num_cores):
            var base = core * Self.MAX_THREADS_PER_CORE
            for thread in range(threads_per_core):
                thread_mask.set(base + thread)
        self = Self(thread_mask^)

    def __init__(out self, *, core_mask: BitSet[Self.MAX_CORE_COUNT]):
        """Initializes the execution config with the given core mask.

        Enables every thread of each core set in `core_mask`.

        Args:
            core_mask: The mask of cores to enable for the kernel launch.
        """
        var thread_mask = BitSet[Self.MAX_THREAD_COUNT]()
        for core in range(Self.MAX_CORE_COUNT):
            if core_mask.test(core):
                var base = core * Self.MAX_THREADS_PER_CORE
                for thread in range(Self.MAX_THREADS_PER_CORE):
                    thread_mask.set(base + thread)
        self = Self(thread_mask^)

    def get_near_compute_scratchpad_usage(self) -> Int:
        """Gets the near-compute scratchpad usage configuration.

        Returns:
            The amount of scratchpad used in bytes.
        """
        return self.scratchpad_usage

    def set_near_compute_scratchpad_usage(mut self, var usage: Int):
        """Sets the near-compute scratchpad usage configuration.

        Args:
            usage: The amount of scratchpad to use in bytes.
        """
        self.scratchpad_usage = usage

    def get_num_threads(self) -> Int:
        """Gets the number of threads to use for the kernel launch.

        Returns:
            The number of enabled threads.
        """
        return len(self.thread_mask)

    def set_num_threads(mut self, var num_threads: Int):
        """Sets the number of threads to use for the kernel launch.

        Enables the first `num_threads` threads and disables the rest.

        Args:
            num_threads: The number of threads to use.
        """
        self.thread_mask.clear_all()
        for i in range(num_threads):
            self.thread_mask.set(i)

    def set_num_threads_per_core(mut self, var num_threads_per_core: Int):
        """Sets the number of threads per core to use for the kernel launch.

        For every core that currently has at least one enabled thread, enables
        exactly its first `num_threads_per_core` threads (clamped to
        `MAX_THREADS_PER_CORE`) and disables the rest.

        Args:
            num_threads_per_core: The number of threads per core to use.
        """
        var active = self.get_core_mask()
        var threads_per_core = min(
            num_threads_per_core, Self.MAX_THREADS_PER_CORE
        )
        self.thread_mask.clear_all()
        for core in range(Self.MAX_CORE_COUNT):
            if active.test(core):
                var base = core * Self.MAX_THREADS_PER_CORE
                for thread in range(threads_per_core):
                    self.thread_mask.set(base + thread)

    def get_num_cores(self) -> Int:
        """Gets the number of cores to use for the kernel launch.

        A core counts as enabled when any of its threads are enabled.

        Returns:
            The number of enabled cores.
        """
        return len(self.get_core_mask())

    def set_num_cores(mut self, var num_cores: Int):
        """Sets the number of cores to use for the kernel launch.

        Enables every thread of the first `num_cores` cores and disables the
        rest.

        Args:
            num_cores: The number of cores to use.
        """
        self.thread_mask.clear_all()
        for i in range(num_cores * Self.MAX_THREADS_PER_CORE):
            self.thread_mask.set(i)

    def request_all_cores(mut self):
        """Requests all cores to be used for the kernel launch."""
        self.thread_mask.set_all()

    def get_core_mask(self) -> BitSet[Self.MAX_CORE_COUNT]:
        """Gets the core mask to use for the kernel launch.

        A core's bit is set when any of its threads are enabled.

        Returns:
            The core mask derived from the thread mask.
        """
        var core_mask = BitSet[Self.MAX_CORE_COUNT]()
        for core in range(Self.MAX_CORE_COUNT):
            var base = core * Self.MAX_THREADS_PER_CORE
            for thread in range(Self.MAX_THREADS_PER_CORE):
                if self.thread_mask.test(base + thread):
                    core_mask.set(core)
                    break
        return core_mask^

    def set_core_mask(mut self, var core_mask: BitSet[Self.MAX_CORE_COUNT]):
        """Sets the core mask to use for the kernel launch.

        Enables every thread of each core set in `core_mask` and disables the
        threads of every other core.

        Args:
            core_mask: The core mask to use.
        """
        self.thread_mask.clear_all()
        for core in range(Self.MAX_CORE_COUNT):
            if core_mask.test(core):
                var base = core * Self.MAX_THREADS_PER_CORE
                for thread in range(Self.MAX_THREADS_PER_CORE):
                    self.thread_mask.set(base + thread)

    def get_thread_mask(self) -> BitSet[Self.MAX_THREAD_COUNT]:
        """Gets the thread mask to use for the kernel launch.

        Returns:
            The thread mask to use.
        """
        return self.thread_mask.copy()

    def set_thread_mask(
        mut self, var thread_mask: BitSet[Self.MAX_THREAD_COUNT]
    ):
        """Sets the thread mask to use for the kernel launch.

        Args:
            thread_mask: The thread mask to use.
        """
        self.thread_mask = thread_mask^

    def request_all_threads(mut self):
        """Requests all threads to be used for the kernel launch."""
        self.thread_mask.set_all()

    def get_matrix_unit_usage(self) -> Bool:
        """Gets the matrix unit usage configuration.

        Returns:
            Whether to use matrix units.
        """
        return self.enable_matrix_unit

    def set_matrix_unit_usage(mut self, var usage: Bool):
        """Sets the matrix unit usage configuration.

        Args:
            usage: Whether to use matrix units.
        """
        self.enable_matrix_unit = usage
