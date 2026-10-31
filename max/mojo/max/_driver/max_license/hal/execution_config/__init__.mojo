# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.collections import BitSet
from max.gpu.host import ConstantMemoryMapping, Dim, LaunchAttribute
from max.gpu.host import _DumpPath
from max.gpu.host import ExecutionConfig
from .gpu_execution_config import GPUExecutionConfiguration


trait AsmDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump kernel ASM to a path.
    """

    comptime asm_dump_path: _DumpPath = False
    """The output path used for ASM dumps."""


trait LlvmDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump LLVM IR to a path."""

    comptime llvm_dump_path: _DumpPath = False
    """The output path used for LLVM IR dumps."""


# NOTE: This is where PTX goes, SASS is ASM for NVIDIA.
trait VendorIRDumpableExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to dump vendor-specific IR to a path.
    """

    comptime vendor_ir_dump_path: _DumpPath = False
    """The output path used for vendor-specific IR dumps."""


trait ConstantMemoryMappingExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to hold constant memory
    mappings."""

    def get_constant_memory_mappings[
        o: ImmOrigin
    ](ref[o] self) -> Span[ConstantMemoryMapping, o]:
        """Gets the constant memory mappings configured for the execution config.

        Returns:
            The constant memory mappings.
        """
        ...

    def set_constant_memory_mappings(
        mut self,
        var mappings: List[ConstantMemoryMapping],
    ):
        """Sets the constant memory mappings for the execution config.

        Args:
            mappings: The constant memory mappings to apply.
        """
        ...


trait CoresExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to specify how many cores to use for
    the kernel launch. The meaning of "core" is implementation defined.

    Vector ALUs are **NOT** considered cores, on GPUs this maps more closely to SMs,
    and on CPUs this maps more closely to physical cores.
    """

    comptime MAX_CORE_COUNT: Int

    def __init__(out self, *, num_cores: Int):
        """Initializes the execution config with the given number of cores.

        Args:
            num_cores: The number of cores to use for the kernel launch.
        """
        ...

    def get_num_cores(self) -> Int:
        """Gets the number of cores to use for the kernel launch.

        Returns:
            The number of cores to use.
        """
        ...

    def set_num_cores(mut self, var num_cores: Int):
        """Sets the number of cores to use for the kernel launch.

        Args:
            num_cores: The number of cores to use.
        """
        ...

    def request_all_cores(mut self):
        """Requests all cores to be used for the kernel launch."""
        ...


trait ParticularCoresExecutionConfig(CoresExecutionConfig):
    """An `ExecutionConfig` which has the ability to specify which cores to use for the kernel launch.

    The meaning of "core" is implementation defined.

    The indexing scheme for cores is implementation defined.

    Vector ALUs are **NOT** considered cores, on GPUs this maps more closely to SMs,
    and on CPUs this maps more closely to physical cores.
    """

    def __init__(out self, *, core_mask: BitSet[Self.MAX_CORE_COUNT]):
        """Initializes the execution config with the given core mask.

        Args:
            core_mask: The core mask to use for the kernel launch.
        """
        ...

    def __init__[
        bit_mask_size: Int, //
    ](out self, *, core_mask: BitSet[bit_mask_size]):
        """Initializes the execution config with the given core mask.

        Args:
            core_mask: The core mask to use for the kernel launch.
        """
        self = Self(
            core_mask=BitSet[Self.MAX_CORE_COUNT](resized_from=core_mask)
        )

    def get_core_mask(self) -> BitSet[Self.MAX_CORE_COUNT]:
        """Gets the core mask to use for the kernel launch.

        Returns:
            The core mask to use.
        """
        ...

    def set_core_mask(mut self, var core_mask: BitSet[Self.MAX_CORE_COUNT]):
        """Sets the core mask to use for the kernel launch.

        Args:
            core_mask: The core mask to use.
        """
        ...

    def set_core_mask[
        bit_mask_size: Int, //
    ](mut self, var core_mask: BitSet[bit_mask_size]):
        """Sets the execution config's core mask from a mask of arbitrary size.

        Args:
            core_mask: The core mask to use for the kernel launch.
        """
        self.set_core_mask(BitSet[Self.MAX_CORE_COUNT](resized_from=core_mask))

    def request_all_cores(mut self):
        """Requests all cores to be used for the kernel launch."""
        var core_mask = BitSet[Self.MAX_CORE_COUNT]()
        core_mask.set_all()
        self.set_core_mask(core_mask^)


trait ThreadCountExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to specify how many total threads to use
    for the kernel launch. The meaning of "thread" is implementation defined.

    Vector ALUs are **NOT** considered threads, on GPUs this maps more closely to warps,
    and on CPUs this maps more closely to hardware threads.
    """

    comptime MAX_THREADS_PER_CORE: Int
    comptime MAX_THREAD_COUNT: Int

    def __init__(out self, *, num_threads: Int):
        """Initializes the execution config with the given number of threads.

        Args:
            num_threads: The number of threads to use for the kernel launch.
        """
        ...

    def get_num_threads(self) -> Int:
        """Gets the number of threads to use for the kernel launch.

        Returns:
            The number of threads to use.
        """
        ...

    def set_num_threads(mut self, var num_threads: Int):
        """Sets the number of threads to use for the kernel launch.

        Args:
            num_threads: The number of threads to use.
        """
        ...


trait ThreadsPerCoreExecutionConfig(
    CoresExecutionConfig, ThreadCountExecutionConfig
):
    """An `ExecutionConfig` which has the ability to specify how many threads per core to use
    for the kernel launch. The meaning of "thread" is implementation defined.

    Vector ALUs are **NOT** considered threads, on GPUs this maps more closely to warps,
    and on CPUs this maps more closely to hardware threads.
    """

    def __init__(out self, *, num_cores: Int, num_threads_per_core: Int):
        """Initializes the execution config with the given number of threads per core.

        Args:
            num_cores: The number of cores to use for the kernel launch.
            num_threads_per_core: The number of threads per core to use for the kernel launch.
        """
        ...

    def set_num_threads_per_core(mut self, var num_threads_per_core: Int):
        """Sets the number of threads per core to use for the kernel launch for all cores.

        Args:
            num_threads_per_core: The number of threads per core to use.
        """
        ...


trait ThreadMaskExecutionConfig(
    ParticularCoresExecutionConfig, ThreadCountExecutionConfig
):
    """An `ExecutionConfig` which has the ability to specify which threads to use for
    the kernel launch. The meaning of "thread" is implementation defined.

    Vector ALUs are **NOT** considered threads, on GPUs this maps more closely to warps,
    and on CPUs this maps more closely to hardware threads.
    """

    def __init__(out self, *, thread_mask: BitSet[Int(Self.MAX_THREAD_COUNT)]):
        """Initializes the execution config with the given thread mask.

        Args:
            thread_mask: The thread mask to use for the kernel launch.
        """
        ...

    def __init__[
        bit_mask_size: Int, //
    ](out self, *, thread_mask: BitSet[Int(bit_mask_size)]):
        """Initializes the execution config with the given thread mask.

        Args:
            thread_mask: The thread mask to use for the kernel launch.
        """
        self = Self(
            thread_mask=BitSet[Int(Self.MAX_THREAD_COUNT)](
                resized_from=thread_mask
            )
        )

    def get_thread_mask(self) -> BitSet[Int(Self.MAX_THREAD_COUNT)]:
        """Gets the thread mask to use for the kernel launch.

        Returns:
            The thread mask to use.
        """
        ...

    def set_thread_mask(
        mut self, var thread_mask: BitSet[Int(Self.MAX_THREAD_COUNT)]
    ):
        """Sets the thread mask to use for the kernel launch.

        Args:
            thread_mask: The thread mask to use.
        """
        ...

    def set_thread_mask[
        bit_mask_size: Int, //
    ](mut self, var thread_mask: BitSet[Int(bit_mask_size)]):
        """Sets the execution config's thread mask from a mask of arbitrary size.

        Args:
            thread_mask: The thread mask to use for the kernel launch.
        """
        self.set_thread_mask(
            BitSet[Int(Self.MAX_THREAD_COUNT)](resized_from=thread_mask)
        )

    def request_all_threads(mut self):
        """Requests all threads to be used for the kernel launch."""
        var thread_mask = BitSet[Int(Self.MAX_THREAD_COUNT)]()
        thread_mask.set_all()
        self.set_thread_mask(thread_mask^)


trait OptionalMatrixUnitExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to specify whether enable
    matrix units (e.g. Intel AMX) before or during kernel launch.
    """

    def get_matrix_unit_usage(self) -> Bool:
        """Gets the matrix unit usage configuration.

        Returns:
            Whether to use matrix units.
        """
        ...

    def set_matrix_unit_usage(mut self, var usage: Bool):
        """Sets the matrix unit usage configuration.

        Args:
            usage: Whether to use matrix units.
        """
        ...


trait LaunchAttributeHolderExecutionConfig(ExecutionConfig):
    """An `ExecutionConfig` which has the ability to hold CUDA launch attributes.
    """

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        """Gets the launch attributes configured for kernel launch.

        Returns:
            The configured launch attributes, or `None` if unset.
        """
        ...

    def set_launch_attributes(mut self, var attributes: List[LaunchAttribute]):
        """Sets launch attributes used for kernel launch.

        Args:
            attributes: Launch attributes to apply, or `None` to clear them.
        """
        ...
