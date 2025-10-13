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
"""Provides low-level GPU intrinsic operations and memory access primitives.

Implements hardware-specific intrinsics that map directly to GPU assembly
instructions, focusing on NVIDIA GPU architectures. Includes:

- Global memory load/store operations with cache control
- Warp-level primitives and synchronization
- Memory fence and barrier operations
- Atomic operations and memory ordering primitives

These low-level primitives should be used carefully as they correspond
directly to hardware instructions and require understanding of the
underlying GPU architecture.
"""

from collections.string.string_slice import get_static_string
from os.atomic import Consistency
from sys import is_amd_gpu, is_gpu, is_nvidia_gpu, size_of, _RegisterPackType
from sys._assembly import inlined_assembly
from sys.info import (
    CompilationTarget,
    _is_sm_9x,
    align_of,
    bit_width_of,
    _cdna_4_or_newer,
)
from sys.intrinsics import llvm_intrinsic, readfirstlane

from memory.unsafe import bitcast

from .memory import AddressSpace, CacheOperation, _int_to_str

# ===-----------------------------------------------------------------------===#
# ldg
# ===-----------------------------------------------------------------------===#


@always_inline
fn ldg[
    dtype: DType, //,
    width: Int = 1,
    *,
    alignment: Int = align_of[SIMD[dtype, width]](),
](x: UnsafePointer[Scalar[dtype]]) -> SIMD[dtype, width]:
    """Load data from global memory through the non-coherent cache.

    This function provides a hardware-accelerated global memory load operation
    that uses the GPU's non-coherent cache (equivalent to CUDA's `__ldg` instruction).
    It optimizes for read-only data access patterns.

    Parameters:
        dtype: The data type to load (must be numeric).
        width: The SIMD vector width for vectorized loads.
        alignment: Memory alignment in bytes. Defaults to natural alignment
            of the SIMD vector dtype.

    Args:
        x: Pointer to global memory location to load from.

    Returns:
        SIMD vector containing the loaded data.

    Note:
        - Uses invariant loads which indicate the memory won't change during kernel execution.
        - Particularly beneficial for read-only texture-like access patterns.
        - May improve performance on memory-bound kernels.
    """
    constrained[dtype.is_numeric(), "the dtype must be numeric"]()
    return x.load[width=width, alignment=alignment, invariant=True]()


# ===-----------------------------------------------------------------------===#
# warpgroup_reg
# ===-----------------------------------------------------------------------===#


fn warpgroup_reg_alloc[count: Int]():
    """Allocates additional registers for the executing warp group.

    Hints to the system to increase per-thread registers owned by the
    executing warp. Requests additional registers to increase the absolute
    per-thread maximum register count from its current value to the specified
    count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs
        - Performance optimization hint that may be ignored by the hardware
        - Pair with `warpgroup_reg_dealloc() when extra registers are no
          longer needed
    """

    constrained[
        count % 8 == 0,
        "count argument to warpgroup_reg_alloc must be in multiples of 8",
    ]()

    constrained[
        24 <= count <= 256,
        "count argument must be within 24 and 256",
    ]()

    @parameter
    if _is_sm_9x():
        inlined_assembly[
            "setmaxnreg.inc.sync.aligned.u32 $0;",
            NoneType,
            constraints="n",
        ](Int32(count))


fn warpgroup_reg_dealloc[count: Int]():
    """Deallocates additional registers for the executing warp group.

    Hints to the system to decrease per-thread registers owned by the
    executing warp. Releases extra registers to reduce the absolute per-thread
    maximum register count from its current value to the specified count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8.
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Performance optimization hint that may be ignored by the hardware.
        - Pair with `warpgroup_reg_alloc()` when extra registers are needed.
    """

    constrained[
        count % 8 == 0,
        "count argument to warpgroup_reg_dealloc must be in multiples of 8",
    ]()

    constrained[
        24 <= count <= 256,
        "count argument must be within 24 and 256",
    ]()

    @parameter
    if _is_sm_9x():
        inlined_assembly[
            "setmaxnreg.dec.sync.aligned.u32 $0;",
            NoneType,
            constraints="n",
        ](Int32(count))


# ===-----------------------------------------------------------------------===#
# lop
# ===-----------------------------------------------------------------------===#


@always_inline
fn lop[lut: Int32](a: Int32, b: Int32, c: Int32) -> Int32:
    """Performs an arbitrary logical operation on 3 inputs using a lookup table.

    Implements a 3-input lookup table (LUT) operation. The result is
    determined by bits in the lookup table value for each input combination.

    Parameters:
        lut: 32-bit lookup table value that defines the logical operation.

    Args:
        a: First input value.
        b: Second input value.
        c: Third input value.

    Returns:
        Result of applying the lookup table operation to the inputs.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps to the LOP3.B32 PTX instruction.
        - Lookup table value determines output for each possible input combo.
    """

    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "lop3.b32 $0, $1, $2, $3, $4;",
            Int32,
            constraints="=r,r,n,n,n",
            has_side_effect=False,
        ](a, b, c, Int32(lut))
    else:
        return CompilationTarget.unsupported_target_error[
            Int32,
            operation="lop",
            note="lop() is only supported when targeting NVIDIA GPUs.",
        ]()


# ===-----------------------------------------------------------------------===#
# permute
# ===-----------------------------------------------------------------------===#


@always_inline
fn byte_permute(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
    """Permutes bytes from two 32-bit integers based on a control mask.

    Selects and rearranges bytes from two source integers based on a control
    mask to create a new 32-bit value.

    Args:
        a: First source integer containing bytes to select from.
        b: Second source integer containing bytes to select from.
        c: Control mask that specifies which bytes to select and their
           positions. Each byte in the mask controls selection/placement of
           one output byte.

    Returns:
        A new 32-bit integer containing the selected and rearranged bytes

    Note:
        Byte selection behavior depends on the GPU architecture:
        - On NVIDIA: Maps to PRMT instruction
        - On AMD: Maps to PERM instruction.
    """
    alias asm = _byte_permute_inst()

    return llvm_intrinsic[asm, UInt32, has_side_effect=False](a, b, c)


fn _byte_permute_inst() -> StaticString:
    @parameter
    if is_nvidia_gpu():
        return "llvm.nvvm.prmt"
    elif is_amd_gpu():
        return "llvm.amdgcn.perm"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation="byte_permute",
        ]()


# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@always_inline
fn mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    unsigned integers.

    Multiplies two 16-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow
    detection.

    Args:
        a: First 16-bit unsigned integer operand.
        b: Second 16-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U16 PTX instruction.
        On others, it performs multiplication using 32-bit arithmetic.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.us", UInt32, has_side_effect=False
        ](a, b)

    var au32 = a.cast[DType.uint32]()
    var bu32 = b.cast[DType.uint32]()
    return au32 * bu32


@always_inline
fn mulhi(a: Int16, b: Int16) -> Int32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    signed integers.

    Multiplies two 16-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 16-bit signed integer operand.
        b: Second 16-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S16 PTX instruction.
        On others, it performs multiplication using 32-bit arithmetic.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.s", Int32, has_side_effect=False
        ](a, b)

    var ai32 = a.cast[DType.int32]()
    var bi32 = b.cast[DType.int32]()
    return ai32 * bi32


@always_inline
fn mulhi(a: UInt32, b: UInt32) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.ui", UInt32, has_side_effect=False
        ](a, b)

    var au64 = a.cast[DType.uint64]()
    var bu64 = b.cast[DType.uint64]()
    return ((au64 * bu64) >> 32).cast[DType.uint32]()


@always_inline
fn mulhi(a: Int32, b: Int32) -> Int32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    signed integers.

    Multiplies two 32-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.i", Int32, has_side_effect=False
        ](a, b)

    var ai64 = a.cast[DType.int64]()
    var bi64 = b.cast[DType.int64]()
    return ((ai64 * bi64) >> 32).cast[DType.int32]()


# ===-----------------------------------------------------------------------===#
# mulwide
# ===-----------------------------------------------------------------------===#


@always_inline
fn mulwide(a: UInt32, b: UInt32) -> UInt64:
    """Performs a wide multiplication of two 32-bit unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The full 64-bit product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.U32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """

    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.u32 $0, $1, $2;",
            UInt64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var au64 = a.cast[DType.uint64]()
    var bu64 = b.cast[DType.uint64]()
    return au64 * bu64


@always_inline
fn mulwide(a: Int32, b: Int32) -> Int64:
    """Performs a wide multiplication of two 32-bit signed integers.

    Multiplies two 32-bit signed integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits or be negative.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The full 64-bit signed product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.S32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """

    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.s32 $0, $1, $2;",
            Int64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var ai64 = a.cast[DType.int64]()
    var bi64 = b.cast[DType.int64]()
    return ai64 * bi64


@always_inline
fn get_ib_sts() -> Int32:
    """Returns the IB status of the current thread.

    Returns:
        The IB status of the current thread.
    """
    if is_amd_gpu():
        return inlined_assembly[
            "s_getreg_b32 $0, hwreg(HW_REG_IB_STS);",
            Int32,
            constraints="=r",
            has_side_effect=False,
        ]()
    else:
        return 0


# ===-----------------------------------------------------------------------===#
# threadfence
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Scope(
    EqualityComparable, Identifiable, ImplicitlyCopyable, Movable, Writable
):
    """Represents memory synchronization scope levels for GPU memory operations.

    Defines different scopes of memory visibility and synchronization, from
    thread-local to system-wide. Each scope level determines how memory
    operations are ordered and visible across different execution units.

    The scope levels form a hierarchy, with each higher level providing
    stronger ordering guarantees but potentially higher synchronization costs.
    """

    var _value: Int

    alias NONE = Self(0)
    """No memory ordering guarantees. Operations may be reordered freely."""

    alias THREAD = Self(1)
    """Thread-level scope. Memory operations are ordered within a single thread."""

    alias WARP = Self(2)
    """Warp-level scope. Memory operations are ordered within a warp of threads."""

    alias BLOCK = Self(3)
    """Block-level scope. Memory operations ordered within a thread block/CTA."""

    alias CLUSTER = Self(4)
    """Cluster-level scope. Memory operations ordered within a thread block cluster."""

    alias GPU = Self(5)
    """GPU-level scope. Memory operations are ordered across all threads on the GPU."""

    alias SYSTEM = Self(6)
    """System-wide scope. Memory operations ordered across the entire system."""

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `Scope` instances are equal.

        Uses pointer comparison for efficiency.

        Args:
            other: The other `Scope` instance to compare with.

        Returns:
            True if the instances are the same, False otherwise.
        """
        return self is other

    fn __is__(self, other: Self) -> Bool:
        """Checks if two `Scope` instances have the same value.

        Compares the underlying integer values.

        Args:
            other: The other `Scope` instance to compare with.

        Returns:
            True if the values are the same, False otherwise.
        """
        return self._value == other._value

    @no_inline
    fn write_to(self, mut w: Some[Writer]):
        """Writes the string representation of the scope to a writer.

        Args:
            w: The writer to write to.
        """
        if self is Self.NONE:
            return w.write("none")
        if self is Self.THREAD:
            return w.write("thread")
        if self is Self.WARP:
            return w.write("warp")
        if self is Self.BLOCK:
            return w.write("block")
        if self is Self.CLUSTER:
            return w.write("cluster")
        if self is Self.GPU:
            return w.write("gpu")
        if self is Self.SYSTEM:
            return w.write("system")

        return w.write("<<unknown scope>>")

    @no_inline
    fn __str__(self) -> String:
        """Returns the string representation of the memory scope.

        Returns:
            A string representation of the memory scope.
        """
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Returns the string representation of the memory scope.

        Returns:
            A string representation of the memory scope.
        """
        return String("Scope(", self, ")")

    @always_inline("nodebug")
    fn mnemonic(self) -> StaticString:
        """Returns the mnemonic string representation of the memory scope.

        Converts the memory scope level into a string mnemonic used by LLVM/NVVM
        intrinsics for memory operations.

        Returns:
            A string literal containing the mnemonic.
        """
        if self in (Self.NONE, Self.THREAD, Self.WARP):
            return ""
        if self is Self.BLOCK:
            return "cta"
        if self is Self.CLUSTER:
            return "cluster"
        if self is Self.GPU:
            return "gpu"
        if self is Self.SYSTEM:
            return "sys"
        return "<<invalid scope>>"


@always_inline
fn threadfence[scope: Scope = Scope.GPU]():
    """Enforces ordering of memory operations across threads.

    Acts as a memory fence/barrier that ensures all memory operations (both
    loads and stores) issued before the fence are visible to other threads
    within the specified scope before any memory operations after the fence.

    Parameters:
        scope: Memory scope level for the fence. Defaults to GPU-wide scope.
              Valid values are:
              - Scope.BLOCK: Orders memory within a thread block/CTA.
              - Scope.GPU: Orders memory across all threads on the GPU (default).
              - Scope.SYSTEM: Orders memory across the entire system.

    Note:
        - Maps directly to CUDA `__threadfence()` family of functions.
        - Critical for synchronizing memory access in parallel algorithms.
        - Performance impact increases with broader scopes.
    """
    constrained[
        scope in (Scope.GPU, Scope.BLOCK, Scope.SYSTEM),
        "invalid threadfence scope",
    ]()
    constrained[
        is_nvidia_gpu(), "threadfence is only implemented on NVIDIA GPUs"
    ]()

    alias suffix = "gl" if scope is Scope.GPU else scope.mnemonic()
    llvm_intrinsic["llvm.nvvm.membar." + suffix, NoneType]()


# ===-----------------------------------------------------------------------===#
# release / acquire
# ===-----------------------------------------------------------------------===#


fn _get_type_suffix[dtype: DType]() -> StaticString:
    alias str = get_static_string["u", _int_to_str[dtype.bit_width()]()]()
    return str


fn _get_nvtx_register_constraint[dtype: DType]() -> StaticString:
    constrained[
        is_nvidia_gpu(),
        (
            "the _get_nvtx_register_constraint function is currently restricted"
            " to only be defined on NVIDIA GPUs"
        ),
    ]()
    if dtype is DType.bool:
        return "b"
    if dtype.is_half_float():
        return "h"
    if dtype.is_integral():
        alias width = dtype.bit_width()
        if width == 16:
            return "c"
        if width == 32:
            return "r"
        if width == 64:
            return "l"
    if dtype is DType.float32:
        return "f"
    if dtype is DType.float64:
        return "d"

    return "<<unknown_register_constraint>>"


fn _get_nvtx_pointer_constraint() -> StaticString:
    constrained[
        is_nvidia_gpu(),
        (
            "the _get_nvtx_pointer_constraint function is currently restricted"
            " to only be defined on NVIDIA GPUs"
        ),
    ]()
    return _get_nvtx_register_constraint[DType.int]()


@always_inline
fn store_release[
    dtype: DType, //,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[Scalar[dtype], **_], value: Scalar[dtype]):
    """Performs an atomic store with release memory ordering semantics.

    This function provides a memory barrier that ensures all previous memory operations
    from the calling thread are visible to other threads before this store is performed.

    Parameters:
        dtype: The data type to store.
        scope: Memory scope for the operation (default: Scope.SYSTEM).
        memory: Whether to include memory side effects in constraints (default: True).
        alignment: The alignment of the data.

    Args:
        ptr: Pointer to the memory location to store to.
        value: Value to store.

    Note:
        - Only supported on GPUs.
        - Maps directly to PTX st.release instruction on NVIDIA, LLVM atomic
          store on AMDGPU.
        - Ensures all previous memory operations complete before this store.
        - Critical for implementing synchronization primitives.
    """
    constrained[is_gpu(), "atomic store only supported on GPU"]()

    @parameter
    if is_nvidia_gpu():
        alias mem_constraint = StaticString(",~{memory}") if memory else ""
        alias constraints = _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        alias scope_str = scope.mnemonic()
        inlined_assembly[
            "st.release."
            + ((scope_str + ".") if scope_str else "")
            + "global."
            + _get_type_suffix[dtype]()
            + " [$1], $0;",
            NoneType,
            constraints=constraints,
        ](value, ptr)
    elif is_amd_gpu():
        __mlir_op.`pop.store`[
            alignment = alignment._mlir_value,
            ordering = Consistency.RELEASE.__mlir_attr(),
        ](value, ptr.address)
    else:
        return CompilationTarget.unsupported_target_error[
            operation="store_release"
        ]()


@always_inline
fn load_acquire[
    dtype: DType, //,
    *,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[Scalar[dtype], **_]) -> Scalar[dtype]:
    """Performs an atomic load operation with acquire memory ordering semantics.

    This function provides a memory barrier that ensures no subsequent memory operations
    from the calling thread are executed until after this load completes.

    Parameters:
        dtype: The data type to load.
        scope: Memory scope for the operation (default: Scope.SYSTEM).
        memory: Whether to include memory side effects in constraints (default: True).
        alignment: The alignment of the pointer.

    Args:
        ptr: Pointer to the memory location to load from.

    Returns:
        The loaded value.

    Note:
        - Only supported on GPUs.
        - Maps directly to PTX ld.acquire instruction on NVIDIA, LLVM atomic
          load on AMDGPU.
        - Ensures subsequent memory operations don't execute until after load.
        - Critical for implementing synchronization primitives.
    """
    constrained[is_gpu(), "atomic load only supported on GPU"]()

    @parameter
    if is_nvidia_gpu():
        alias mem_constraint = StaticString(",~{memory}") if memory else ""
        alias constraints = "=" + _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        alias scope_str = scope.mnemonic()
        return inlined_assembly[
            "ld.acquire."
            + ((scope_str + ".") if scope_str else "")
            + "global."
            + _get_type_suffix[dtype]()
            + " $0, [$1];",
            Scalar[dtype],
            constraints=constraints,
        ](ptr.address_space_cast[AddressSpace.GENERIC]())
    elif is_amd_gpu():
        return __mlir_op.`pop.load`[
            alignment = alignment._mlir_value,
            ordering = Consistency.ACQUIRE.__mlir_attr(),
        ](ptr.address)
    else:
        return CompilationTarget.unsupported_target_error[
            Scalar[dtype],
            operation="load_acquire",
        ]()


@always_inline
fn store_volatile[
    dtype: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[dtype], **_], value: Scalar[dtype]):
    """Performs a volatile store operation that cannot be optimized away.

    This function guarantees that the store operation will be performed exactly as
    specified, without being reordered or optimized away by the compiler.

    Parameters:
        dtype: The data type to store.
        memory: Whether to include memory side effects in constraints (default: True).

    Args:
        ptr: Pointer to the memory location to store to.
        value: Value to store.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps directly to PTX st.volatile instruction.
        - Prevents compiler optimization of the store operation.
        - Useful for memory-mapped I/O or synchronization primitives.
        - May have performance implications compared to regular stores.
    """
    constrained[
        is_nvidia_gpu(), "store_volatile is not currently supported on AMD GPUs"
    ]()
    alias mem_constraint = StaticString(",~{memory}") if memory else ""
    alias constraints = _get_nvtx_register_constraint[
        dtype
    ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
    inlined_assembly[
        "st.volatile.global." + _get_type_suffix[dtype]() + " [$1], $0;",
        NoneType,
        constraints=constraints,
    ](value, ptr.address_space_cast[AddressSpace.GENERIC]())


@always_inline
fn load_volatile[
    dtype: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[dtype], **_]) -> Scalar[dtype]:
    """Performs a volatile load operation that cannot be optimized away.

    This function guarantees that the load operation will be performed exactly as
    specified, without being reordered or optimized away by the compiler.

    Parameters:
        dtype: The data type to load.
        memory: Whether to include memory side effects in constraints (default: True).

    Args:
        ptr: Pointer to the memory location to load from.

    Returns:
        The loaded value.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps directly to PTX ld.volatile instruction.
        - Prevents compiler optimization of the load operation.
        - Useful for memory-mapped I/O or synchronization primitives.
        - May have performance implications compared to regular loads.
    """
    constrained[
        is_nvidia_gpu(), "load_volatile is not currently supported on AMD GPUs"
    ]()
    alias mem_constraint = StaticString(",~{memory}") if memory else ""
    alias constraints = "=" + _get_nvtx_register_constraint[
        dtype
    ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
    return inlined_assembly[
        "ld.volatile.global." + _get_type_suffix[dtype]() + " $0, [$1];",
        Scalar[dtype],
        constraints=constraints,
    ](ptr.address_space_cast[AddressSpace.GENERIC]())


@register_passable("trivial")
struct AMDBufferResource:
    var desc: SIMD[DType.uint32, 4]
    """128-bit descriptor for a buffer resource on AMD GPUs. Used for
    buffer_load/buffer_store instructions."""

    @always_inline("nodebug")
    fn __init__[
        dtype: DType
    ](
        out self,
        gds_ptr: UnsafePointer[Scalar[dtype], **_],
        num_records: Int = Int(UInt32.MAX),
    ):
        constrained[
            is_amd_gpu(),
            (
                "The AMDBufferResource struct is only applicable on AMDGPU"
                " hardware."
            ),
        ]()

        self.desc = SIMD[DType.uint32, 4](0)
        var address = bitcast[DType.uint32, 2](UInt64(Int(gds_ptr)))
        self.desc[0] = address[0]
        # assuming 0 stride currently
        self.desc[1] = address[1]
        self.desc[2] = size_of[dtype]() * num_records
        # https://github.com/ROCm/composable_kernel/blob/3b2302081eab4975370e29752343058392578bcb/include/ck/ck.hpp#L84
        self.desc[3] = 0x00020000

    @always_inline("nodebug")
    fn __init__(out self):
        constrained[
            is_amd_gpu(),
            (
                "The AMDBufferResource struct is only applicable on AMDGPU"
                " hardware."
            ),
        ]()
        self.desc = 0

    @always_inline("nodebug")
    fn get_base_ptr(self) -> Int:
        return Int(
            bitcast[DType.int64, 1](
                SIMD[DType.uint32, 2](self.desc[0], self.desc[1])
            )
        )

    @always_inline("nodebug")
    fn load[
        dtype: DType,
        width: Int,
        *,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
    ](
        self,
        vector_offset: Int32,
        *,
        scalar_offset: Int32 = 0,
    ) -> SIMD[
        dtype, width
    ]:
        constrained[
            is_amd_gpu(),
            "The buffer_load function is only applicable on AMDGPU hardware.",
        ]()

        alias bytes = size_of[dtype]() * width
        alias aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * size_of[dtype]()
        var scalar_offset_bytes = scalar_offset * size_of[dtype]()

        var load_val = llvm_intrinsic[
            "llvm.amdgcn.raw.buffer.load",
            SIMD[
                _get_buffer_intrinsic_simd_dtype[bytes](),
                _get_buffer_intrinsic_simd_width[bytes](),
            ],
            has_side_effect=False,
        ](self.desc, vector_offset_bytes, scalar_offset_bytes, aux)

        return bitcast[dtype, width](load_val)

    @always_inline("nodebug")
    fn load_to_lds[
        dtype: DType,
        *,
        width: Int = 1,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
    ](
        self,
        vector_offset: Int32,
        shared_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.SHARED
        ],
        *,
        scalar_offset: Int32 = 0,
    ):
        """Loads data from global memory and stores to shared memory.

        Copies from global memory to shared memory (aka LDS) bypassing storing to
        register.

        Parameters:
            dtype: The dtype of the data to be loaded.
            width: The SIMD vector width.
            cache_policy: Cache operation policy controlling cache behavior at all levels.

        Args:
            vector_offset: Vector memory offset in elements (per thread).
            shared_ptr: Shared memory address.
            scalar_offset: Scalar memory offset in elements (shared across wave).
        """
        constrained[
            is_amd_gpu(),
            (
                "The buffer_load_lds function is only applicable on AMDGPU"
                " hardware."
            ),
        ]()

        alias bytes = size_of[dtype]() * width
        alias aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * size_of[dtype]()
        var scalar_offset_bytes = scalar_offset * size_of[dtype]()

        llvm_intrinsic[
            "llvm.amdgcn.raw.buffer.load.lds", NoneType, has_side_effect=True
        ](
            self.desc,
            shared_ptr,
            Int32(bytes),
            vector_offset_bytes,
            scalar_offset_bytes,
            Int32(0),
            aux,
        )

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        width: Int,
        *,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
    ](
        self,
        vector_offset: Int32,
        val: SIMD[dtype, width],
        *,
        scalar_offset: Int32 = 0,
    ):
        """Stores a register variable to global memory with cache operation control.

        Writes to global memory from a register with high-level cache control.

        Parameters:
            dtype: The data type.
            width: The SIMD vector width.
            cache_policy: Cache operation policy controlling cache behavior at all levels.

        Args:
            vector_offset: Vector memory offset in elements (per thread).
            val: Value to write.
            scalar_offset: Scalar memory offset in elements (shared across wave).

        Note:
            - Only supported on AMD GPUs.
            - Provides high-level cache control via CacheOperation enum values.
            - Maps directly to llvm.amdgcn.raw.buffer.store intrinsics.
            - Cache control bits:
            - SC[1:0] controls coherency scope: 0=wave, 1=group, 2=device, 3=system.
            - nt=True: Use streaming-optimized cache policies (recommended for streaming data).
        """
        constrained[
            is_amd_gpu(),
            "The buffer_store function is only applicable on AMDGPU hardware.",
        ]()

        alias bytes = width * size_of[dtype]()
        alias aux: Int32 = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * size_of[dtype]()
        var scalar_offset_bytes = scalar_offset * size_of[dtype]()

        var store_val = bitcast[
            _get_buffer_intrinsic_simd_dtype[bytes](),
            _get_buffer_intrinsic_simd_width[bytes](),
        ](val)

        llvm_intrinsic[
            "llvm.amdgcn.raw.buffer.store", NoneType, has_side_effect=True
        ](
            store_val,
            self.desc,
            vector_offset_bytes,
            scalar_offset_bytes,
            aux,
        )


@parameter
fn _cache_operation_to_amd_aux[cache_policy: CacheOperation]() -> Int32:
    """Converts CacheOperation to AMD auxiliary parameter at compile time.

    Parameters:
        cache_policy: The cache operation policy.

    Returns:
        The auxiliary parameter value formatted for AMD buffer operations.
        Format: bit 0 = SC0, bit 1 = NT, bit 4 = SC1
    """

    @parameter
    if cache_policy is CacheOperation.ALWAYS:
        return 0x00  # SC=00, NT=0
    elif cache_policy is CacheOperation.STREAMING:
        return 0x02  # SC=00, NT=1
    elif cache_policy is CacheOperation.GLOBAL:
        return 0x10  # SC=10, NT=0
    elif cache_policy is CacheOperation.VOLATILE:
        return 0x11  # SC=11, NT=0
    elif cache_policy is CacheOperation.WORKGROUP | CacheOperation.STREAMING:
        return 0x03  # SC=01, NT=1
    elif cache_policy is CacheOperation.GLOBAL | CacheOperation.STREAMING:
        return 0x12  # SC=10, NT=1
    else:
        # Default to ALWAYS for unknown/unsupported operations
        return 0x00

    # Additional cache operations for potential future support:
    # CacheOperation.WORKGROUP -> 0x01 (SC=01, NT=0) - Workgroup/CU-level coherency
    # CacheOperation.GLOBAL_STREAMING -> 0x12 (SC=10, NT=1) - Global + streaming
    # CacheOperation.VOLATILE_STREAMING -> 0x13 (SC=11, NT=1) - Volatile + streaming


fn _get_buffer_intrinsic_simd_dtype[bytes: Int]() -> DType:
    @parameter
    if bytes == 1:
        return DType.uint8
    elif bytes == 2:
        return DType.uint16
    else:
        constrained[bytes in (4, 8, 16), "Width not supported"]()
        return DType.uint32


@parameter
fn _get_buffer_intrinsic_simd_width[bytes: Int]() -> Int:
    return bytes // size_of[DType.uint32]() if bytes >= 4 else 1


# ===-----------------------------------------------------------------------===#
# AMD LDS transpose reads (ds.read.tr*)
# ===-----------------------------------------------------------------------===#


@always_inline
fn ds_read_tr16_b64[
    dtype: DType, //,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED, **_
    ]
) -> SIMD[dtype, 4]:
    """Reads a 64-bit LDS transpose block using TR16 layout and returns SIMD[dtype, 4] of 16-bit types.

    Args:
        shared_ptr: Pointer to the LDS transpose block.

    Returns:
        SIMD[dtype, 4] of 16-bit types.

    Notes:
        - Only supported on AMD GPUs.
        - Maps directly to llvm.amdgcn.ds.read.tr16.b64 intrinsic.
        - Result width is fixed to 4 elements of dtype.
    """

    constrained[
        is_amd_gpu(),
        "The ds_read_tr16_b64 function is only applicable on AMDGPU hardware.",
    ]()

    constrained[
        size_of[dtype]() == 2,
        "ds_read_tr16_b64 supports 16-bit dtypes.",
    ]()

    # constrained[
    #     _cdna_4_or_newer(), "ds_read_tr16_b64 is only supported on CDNA4+"
    # ]()

    return llvm_intrinsic[
        "llvm.amdgcn.ds.read.tr16.b64", SIMD[dtype, 4], has_side_effect=False
    ](shared_ptr)


# ===-----------------------------------------------------------------------===#
# AMD permlane shuffle
# ===-----------------------------------------------------------------------===#


@always_inline
fn permlane_swap[
    dtype: DType, //, stride: Int
](val1: Scalar[dtype], val2: Scalar[dtype]) -> SIMD[dtype, 2]:
    constrained[
        is_amd_gpu(),
        (
            "The _amd_permlane_swap function is only applicable on AMDGPU"
            " hardware."
        ),
    ]()
    # constrained[
    #     _cdna_4_or_newer(), "permlane swap is only supported on CDNA4+"
    # ]()
    constrained[bit_width_of[dtype]() == 32, "Unsupported dtype"]()
    constrained[stride in (16, 32), "Unsupported stride"]()

    alias asm = "llvm.amdgcn.permlane" + String(stride) + ".swap"
    var result = llvm_intrinsic[
        asm,
        _RegisterPackType[Int32, Int32],
        has_side_effect=False,
    ](
        bitcast[DType.int32, 1](val1),
        bitcast[DType.int32, 1](val2),
        False,
        False,
    )
    return SIMD[dtype, 2](
        bitcast[dtype, 1](result[0]), bitcast[dtype, 1](result[1])
    )


fn permlane_shuffle[
    dtype: DType, simd_width: Int, //, stride: Int
](val: SIMD[dtype, simd_width], out res: __type_of(val)):
    var lane_group = lane_id() // stride

    var out = __type_of(res)()

    @parameter
    for i in range(simd_width):
        out[i] = permlane_swap[stride](val[i], val[i])[(lane_group + 1) % 2]
    return out
