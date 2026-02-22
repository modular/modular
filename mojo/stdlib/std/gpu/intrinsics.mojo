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
from ffi import external_call
from sys import (
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    is_apple_gpu,
    size_of,
    _RegisterPackType,
)
from sys._assembly import inlined_assembly
from sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    align_of,
    bit_width_of,
    _cdna_3_or_newer,
    _cdna_4_or_newer,
    _is_amd_rdna1,
    _is_amd_rdna2,
    _is_amd_rdna3,
    _is_amd_rdna4,
)
from sys.intrinsics import llvm_intrinsic, readfirstlane

from memory.unsafe import bitcast

from .memory.memory import CacheOperation, _int_to_str

# ===-----------------------------------------------------------------------===#
# ldg
# ===-----------------------------------------------------------------------===#


@always_inline
fn ldg[
    dtype: DType,
    //,
    width: Int = 1,
    *,
    alignment: Int = align_of[SIMD[dtype, width]](),
](x: UnsafePointer[mut=False, Scalar[dtype]]) -> SIMD[dtype, width]:
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
    comptime assert dtype.is_numeric(), "the dtype must be numeric"
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

    comptime assert (
        count % 8 == 0
    ), "count argument to warpgroup_reg_alloc must be in multiples of 8"

    comptime assert (
        24 <= count <= 256
    ), "count argument must be within 24 and 256"

    comptime if _is_sm_9x_or_newer():
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

    comptime assert (
        count % 8 == 0
    ), "count argument to warpgroup_reg_dealloc must be in multiples of 8"

    comptime assert (
        24 <= count <= 256
    ), "count argument must be within 24 and 256"

    comptime if _is_sm_9x_or_newer():
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

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "lop3.b32 $0, $1, $2, $3, $4;",
            Int32,
            constraints="=r,r,n,n,n",
            has_side_effect=False,
        ](a, b, c, lut)
    else:
        return CompilationTarget.unsupported_target_error[
            Int32,
            operation = __get_current_function_name(),
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
    comptime asm = _byte_permute_inst()

    return llvm_intrinsic[asm, UInt32, has_side_effect=False](a, b, c)


fn _byte_permute_inst() -> StaticString:
    comptime if is_nvidia_gpu():
        return "llvm.nvvm.prmt"
    elif is_amd_gpu():
        return "llvm.amdgcn.perm"
    else:
        return CompilationTarget.unsupported_target_error[
            StaticString,
            operation = __get_current_function_name(),
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

    comptime if is_nvidia_gpu():
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

    comptime if is_nvidia_gpu():
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

    comptime if is_nvidia_gpu():
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

    comptime if is_nvidia_gpu():
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

    comptime if is_nvidia_gpu():
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

    comptime if is_nvidia_gpu():
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
struct Scope(Equatable, ImplicitlyCopyable, Writable):
    """Represents memory synchronization scope levels for GPU memory operations.

    Defines different scopes of memory visibility and synchronization, from
    thread-local to system-wide. Each scope level determines how memory
    operations are ordered and visible across different execution units.

    The scope levels form a hierarchy, with each higher level providing
    stronger ordering guarantees but potentially higher synchronization costs.
    """

    var _value: Int

    comptime NONE = Self(0)
    """No memory ordering guarantees. Operations may be reordered freely."""

    comptime THREAD = Self(1)
    """Thread-level scope. Memory operations are ordered within a single thread."""

    comptime WARP = Self(2)
    """Warp-level scope. Memory operations are ordered within a warp of threads."""

    comptime BLOCK = Self(3)
    """Block-level scope. Memory operations ordered within a thread block/CTA."""

    comptime CLUSTER = Self(4)
    """Cluster-level scope. Memory operations ordered within a thread block cluster."""

    comptime GPU = Self(5)
    """GPU-level scope. Memory operations are ordered across all threads on the GPU."""

    comptime SYSTEM = Self(6)
    """System-wide scope. Memory operations ordered across the entire system."""

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two `Scope` instances are equal.

        Uses pointer comparison for efficiency.

        Args:
            other: The other `Scope` instance to compare with.

        Returns:
            True if the instances are the same, False otherwise.
        """
        return self._value == other._value

    @no_inline
    fn write_to(self, mut w: Some[Writer]):
        """Writes the string representation of the scope to a writer.

        Args:
            w: The writer to write to.
        """
        if self == Self.NONE:
            return w.write("none")
        if self == Self.THREAD:
            return w.write("thread")
        if self == Self.WARP:
            return w.write("warp")
        if self == Self.BLOCK:
            return w.write("block")
        if self == Self.CLUSTER:
            return w.write("cluster")
        if self == Self.GPU:
            return w.write("gpu")
        if self == Self.SYSTEM:
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
        if self == Self.BLOCK:
            return "cta"
        if self == Self.CLUSTER:
            return "cluster"
        if self == Self.GPU:
            return "gpu"
        if self == Self.SYSTEM:
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
    comptime assert scope in (
        Scope.GPU,
        Scope.BLOCK,
        Scope.SYSTEM,
    ), "invalid threadfence scope"
    comptime assert (
        is_nvidia_gpu()
    ), "threadfence is only implemented on NVIDIA GPUs"

    comptime suffix = "gl" if scope == Scope.GPU else scope.mnemonic()
    llvm_intrinsic["llvm.nvvm.membar." + suffix, NoneType]()


# ===-----------------------------------------------------------------------===#
# release / acquire
# ===-----------------------------------------------------------------------===#


fn _get_type_suffix[dtype: DType]() -> StaticString:
    comptime str = get_static_string[
        "u", _int_to_str[bit_width_of[dtype]()]()
    ]()
    return str


fn _get_air_atomic_suffix[dtype: DType]() -> StaticString:
    comptime if dtype == DType.float32:
        return "f32"
    elif dtype in (DType.int32, DType.uint32):
        return "i32"
    else:
        comptime assert False, "unsupported dtype for air atomic intrinsics"
        return ""


fn _get_nvtx_register_constraint[dtype: DType]() -> StaticString:
    comptime assert is_nvidia_gpu(), (
        "the _get_nvtx_register_constraint function is currently restricted"
        " to only be defined on NVIDIA GPUs"
    )
    if dtype == DType.bool:
        return "b"
    if dtype.is_half_float():
        return "h"
    if dtype.is_integral():
        comptime width = bit_width_of[dtype]()
        if width == 16:
            return "c"
        if width == 32:
            return "r"
        if width == 64:
            return "l"
    if dtype == DType.float32:
        return "f"
    if dtype == DType.float64:
        return "d"

    return "<<unknown_register_constraint>>"


fn _get_nvtx_pointer_constraint() -> StaticString:
    comptime assert is_nvidia_gpu(), (
        "the _get_nvtx_pointer_constraint function is currently restricted"
        " to only be defined on NVIDIA GPUs"
    )
    return _get_nvtx_register_constraint[DType.int]()


struct _AirMemFlags:
    """AIR memory domain flags used by Apple/Metal intrinsics.
    These values select **which address space's visibility** a fence operates on.
    """

    comptime Device = Int32(1)
    comptime ThreadGroup = Int32(2)


struct _AirScope:
    """AIR synchronization scope for ordering and visibility.
    The scope determines **which set of threads** participates in the ordering
    established by a fence or an atomic op with scope.
    """

    comptime Workgroup = Int32(1)
    comptime Device = Int32(2)
    comptime SIMDGroup = Int32(4)


struct _AirMemOrder:
    """AIR memory ordering semantics for atomic operations and fences."""

    comptime Relaxed = Int32(0)
    comptime SeqCst = Int32(5)


@always_inline
fn store_release[
    dtype: DType,
    //,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...], value: Scalar[dtype]):
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
    comptime assert is_gpu(), "atomic store only supported on GPU"

    comptime if is_nvidia_gpu():
        comptime mem_constraint = StaticString(",~{memory}") if memory else ""
        comptime constraints = _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        comptime scope_str = scope.mnemonic()
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
    elif is_apple_gpu():
        comptime mem_flags = _AirMemFlags.ThreadGroup if ptr.address_space == AddressSpace.SHARED else _AirMemFlags.Device
        comptime air_scope = _AirScope.Workgroup if scope == Scope.BLOCK else _AirScope.Device
        external_call["air.atomic.fence", NoneType](
            mem_flags,
            _AirMemOrder.SeqCst,
            air_scope,
        )
        comptime addr_space = AddressSpace.GLOBAL if ptr.address_space == AddressSpace.GENERIC else ptr.address_space
        comptime store_intrin_base = "air.atomic.local.store" if addr_space == AddressSpace.SHARED else "air.atomic.global.store"
        comptime store_intrin = store_intrin_base + "." + _get_air_atomic_suffix[
            dtype
        ]()
        external_call[store_intrin, NoneType](
            ptr.address_space_cast[addr_space](),
            value,
            _AirMemOrder.Relaxed,
            air_scope,
            True,
        )
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn store_relaxed[
    dtype: DType,
    //,
    *,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...], value: Scalar[dtype]):
    """Performs an atomic store with relaxed memory ordering semantics.

    On NVIDIA, maps to PTX st.relaxed; on AMD, maps to POP atomic store with MONOTONIC ordering.

    Parameters:
        dtype: Data type of the value to store.
        scope: Memory scope for the atomic operation.
        memory: Whether to add memory clobber constraint.
        alignment: Alignment requirement for the pointer.

    Args:
        ptr: Pointer to the memory location.
        value: Value to store.
    """
    comptime assert is_gpu(), "atomic store only supported on GPU"

    comptime if is_nvidia_gpu():
        comptime mem_constraint = StaticString(",~{memory}") if memory else ""
        comptime constraints = _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        comptime scope_str = scope.mnemonic()
        inlined_assembly[
            "st.relaxed."
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
            ordering = Consistency.MONOTONIC.__mlir_attr(),
        ](value, ptr.address)
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn load_acquire[
    dtype: DType,
    //,
    *,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...]) -> Scalar[dtype]:
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
    comptime assert is_gpu(), "atomic load only supported on GPU"

    comptime if is_nvidia_gpu():
        comptime mem_constraint = StaticString(",~{memory}") if memory else ""
        comptime constraints = "=" + _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        comptime scope_str = scope.mnemonic()
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
    elif is_apple_gpu():
        comptime addr_space = AddressSpace.GLOBAL if ptr.address_space == AddressSpace.GENERIC else ptr.address_space
        comptime mem_flags = _AirMemFlags.ThreadGroup if addr_space == AddressSpace.SHARED else _AirMemFlags.Device
        comptime air_scope = _AirScope.Workgroup if scope == Scope.BLOCK else _AirScope.Device
        comptime load_intrin_base = "air.atomic.local.load" if addr_space == AddressSpace.SHARED else "air.atomic.global.load"
        comptime load_intrin = load_intrin_base + "." + _get_air_atomic_suffix[
            dtype
        ]()
        var value = external_call[load_intrin, Scalar[dtype]](
            ptr.address_space_cast[addr_space](),
            _AirMemOrder.Relaxed,
            air_scope,
            True,
        )
        external_call["air.atomic.fence", NoneType](
            mem_flags,
            _AirMemOrder.SeqCst,
            air_scope,
        )
        return value
    else:
        return CompilationTarget.unsupported_target_error[
            Scalar[dtype],
            operation = __get_current_function_name(),
        ]()


@always_inline
fn load_relaxed[
    dtype: DType,
    //,
    *,
    scope: Scope = Scope.SYSTEM,
    memory: Bool = True,
    alignment: Int = align_of[Scalar[dtype]](),
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...]) -> Scalar[dtype]:
    """Performs an atomic load with relaxed memory ordering semantics.

    On NVIDIA, maps to PTX ld.relaxed; on AMD, maps to POP atomic load with MONOTONIC ordering.

    Parameters:
        dtype: Data type of the value to load.
        scope: Memory scope for the atomic operation.
        memory: Whether to add memory clobber constraint.
        alignment: Alignment requirement for the pointer.

    Args:
        ptr: Pointer to the memory location.

    Returns:
        The loaded value.
    """
    comptime assert is_gpu(), "atomic load only supported on GPU"

    comptime if is_nvidia_gpu():
        comptime mem_constraint = StaticString(",~{memory}") if memory else ""
        comptime constraints = "=" + _get_nvtx_register_constraint[
            dtype
        ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
        comptime scope_str = scope.mnemonic()
        return inlined_assembly[
            "ld.relaxed."
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
            ordering = Consistency.MONOTONIC.__mlir_attr(),
        ](ptr.address)
    else:
        return CompilationTarget.unsupported_target_error[
            Scalar[dtype],
            operation = __get_current_function_name(),
        ]()


@always_inline
fn store_volatile[
    dtype: DType, //, memory: Bool = True
](ptr: UnsafePointer[mut=True, Scalar[dtype], ...], value: Scalar[dtype]):
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
    comptime assert (
        is_nvidia_gpu()
    ), "store_volatile is not currently supported on AMD GPUs"
    comptime mem_constraint = StaticString(",~{memory}") if memory else ""
    comptime constraints = _get_nvtx_register_constraint[
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
](ptr: UnsafePointer[mut=False, Scalar[dtype], ...]) -> Scalar[dtype]:
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
    comptime assert (
        is_nvidia_gpu()
    ), "load_volatile is not currently supported on AMD GPUs"
    comptime mem_constraint = StaticString(",~{memory}") if memory else ""
    comptime constraints = "=" + _get_nvtx_register_constraint[
        dtype
    ]() + "," + _get_nvtx_pointer_constraint() + mem_constraint
    return inlined_assembly[
        "ld.volatile.global." + _get_type_suffix[dtype]() + " $0, [$1];",
        Scalar[dtype],
        constraints=constraints,
    ](ptr.address_space_cast[AddressSpace.GENERIC]())


struct AMDBufferResource(TrivialRegisterPassable):
    """128-bit descriptor for a buffer resource on AMD GPUs.

    Used for buffer_load/buffer_store instructions.
    """

    var desc: SIMD[DType.uint32, 4]
    """The 128-bit buffer descriptor encoded as four 32-bit values."""

    @always_inline("nodebug")
    fn __init__[
        dtype: DType
    ](
        out self,
        # TODO: This should propagate mutability correctly.
        # E.g. only allow AMDBufferResource.store when mutable.
        gds_ptr: UnsafePointer[Scalar[dtype], ...],
        num_records: Int = Int(UInt32.MAX),
    ):
        """Constructs an AMD buffer resource descriptor.

        Parameters:
            dtype: Data type of the buffer elements.

        Args:
            gds_ptr: Pointer to the buffer in global memory.
            num_records: Number of records in the buffer.
        """
        comptime assert (
            is_amd_gpu()
        ), "The AMDBufferResource struct is only applicable on AMDGPU hardware."

        self.desc = SIMD[DType.uint32, 4](0)
        var address = bitcast[DType.uint32, 2](UInt64(Int(gds_ptr)))
        self.desc[0] = address[0]
        # assuming 0 stride currently
        self.desc[1] = address[1]
        self.desc[2] = UInt32(size_of[dtype]() * num_records)

        # Architecture-specific word 3 value for buffer resource.
        # https://github.com/ROCm/composable_kernel/blob/3b2302081eab4975370e29752343058392578bcb/include/ck/ck.hpp#L84
        comptime if _is_amd_rdna3() or _is_amd_rdna4():
            # GFX11/GFX12 (RDNA3/RDNA4)
            self.desc[3] = 0x31004000
        elif _is_amd_rdna1() or _is_amd_rdna2():
            # GFX10.x (RDNA1/RDNA2)
            self.desc[3] = 0x31014000
        else:
            comptime assert _cdna_3_or_newer(), (
                "The AMDBufferResource struct is only defined for CDNA 3+"
                " and RDNA 1-4 GPUs."
            )
            # GFX9 (CDNA/GCN)
            self.desc[3] = 0x00020000

    @always_inline("nodebug")
    fn __init__(out self):
        """Constructs a zeroed AMD buffer resource descriptor."""
        comptime assert (
            is_amd_gpu()
        ), "The AMDBufferResource struct is only applicable on AMDGPU hardware."
        self.desc = 0

    @always_inline("nodebug")
    fn get_base_ptr(self) -> Int:
        """Gets the base pointer address from the buffer resource descriptor.

        Returns:
            The base pointer address as an integer.
        """
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
        """Loads data from the buffer using AMD buffer load intrinsic.

        Parameters:
            dtype: Data type to load.
            width: Number of elements to load.
            cache_policy: Cache operation policy.

        Args:
            vector_offset: Offset in elements from the base pointer.
            scalar_offset: Additional scalar offset in elements.

        Returns:
            SIMD vector containing the loaded data.
        """
        comptime assert (
            is_amd_gpu()
        ), "The buffer_load function is only applicable on AMDGPU hardware."

        comptime bytes = size_of[dtype]() * width
        comptime aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

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
            mut=True, Scalar[dtype], address_space = AddressSpace.SHARED
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
        comptime assert (
            is_amd_gpu()
        ), "The buffer_load_lds function is only applicable on AMDGPU hardware."

        comptime bytes = size_of[dtype]() * width
        comptime aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

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
        comptime assert (
            is_amd_gpu()
        ), "The buffer_store function is only applicable on AMDGPU hardware."

        comptime bytes = width * size_of[dtype]()
        comptime aux: Int32 = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

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

    comptime if cache_policy == CacheOperation.ALWAYS:
        return 0x00  # SC=00, NT=0
    elif cache_policy == CacheOperation.STREAMING:
        return 0x02  # SC=00, NT=1
    elif cache_policy == CacheOperation.GLOBAL:
        return 0x10  # SC=10, NT=0
    elif cache_policy == CacheOperation.VOLATILE:
        return 0x11  # SC=11, NT=0
    elif cache_policy == CacheOperation.WORKGROUP | CacheOperation.STREAMING:
        return 0x03  # SC=01, NT=1
    elif cache_policy == CacheOperation.GLOBAL | CacheOperation.STREAMING:
        return 0x12  # SC=10, NT=1
    else:
        # Default to ALWAYS for unknown/unsupported operations
        return 0x00

    # Additional cache operations for potential future support:
    # CacheOperation.WORKGROUP -> 0x01 (SC=01, NT=0) - Workgroup/CU-level coherency
    # CacheOperation.GLOBAL_STREAMING -> 0x12 (SC=10, NT=1) - Global + streaming
    # CacheOperation.VOLATILE_STREAMING -> 0x13 (SC=11, NT=1) - Volatile + streaming


fn _get_buffer_intrinsic_simd_dtype[bytes: Int]() -> DType:
    comptime if bytes == 1:
        return DType.uint8
    elif bytes == 2:
        return DType.uint16
    else:
        comptime assert bytes in (4, 8, 16), "Width not supported"
        return DType.uint32


@parameter
fn _get_buffer_intrinsic_simd_width[bytes: Int]() -> Int:
    return bytes // size_of[DType.uint32]() if bytes >= 4 else 1


# ===-----------------------------------------------------------------------===#
# AMD LDS transpose reads (ds.read.tr*)
# ===-----------------------------------------------------------------------===#


@always_inline
fn ds_read_tr16_b64[
    dtype: DType,
    //,
](
    shared_ptr: UnsafePointer[
        mut=False, Scalar[dtype], address_space = AddressSpace.SHARED
    ]
) -> SIMD[dtype, 4]:
    """Reads a 64-bit LDS transpose block using TR16 layout and returns SIMD[dtype, 4] of 16-bit types.

    Parameters:
        dtype: Data type of the elements (must be 16-bit type).

    Args:
        shared_ptr: Pointer to the LDS transpose block.

    Returns:
        SIMD[dtype, 4] of 16-bit types.

    Notes:
        - Only supported on AMD GPUs.
        - Maps directly to llvm.amdgcn.ds.read.tr16.b64 intrinsic.
        - Result width is fixed to 4 elements of dtype.
    """

    comptime assert (
        is_amd_gpu()
    ), "The ds_read_tr16_b64 function is only applicable on AMDGPU hardware."

    comptime assert (
        size_of[dtype]() == 2
    ), "ds_read_tr16_b64 supports 16-bit dtypes."

    comptime assert (
        _cdna_4_or_newer()
    ), "ds_read_tr16_b64 is only supported on CDNA4+"

    return llvm_intrinsic[
        "llvm.amdgcn.ds.read.tr16.b64", SIMD[dtype, 4], has_side_effect=True
    ](shared_ptr)


# ===-----------------------------------------------------------------------===#
# AMD permlane shuffle
# ===-----------------------------------------------------------------------===#


@always_inline
fn permlane_swap[
    dtype: DType, //, stride: Int
](val1: Scalar[dtype], val2: Scalar[dtype]) -> SIMD[dtype, 2]:
    """Swaps values between lanes using AMD permlane swap instruction.

    Parameters:
        dtype: Data type of the values (must be 32-bit type).
        stride: Swap stride (must be 16 or 32).

    Args:
        val1: First value to swap.
        val2: Second value to swap.

    Returns:
        SIMD vector containing the swapped values.
    """
    comptime assert (
        is_amd_gpu()
    ), "The _amd_permlane_swap function is only applicable on AMDGPU hardware."
    comptime assert (
        _cdna_4_or_newer()
    ), "permlane swap is only supported on CDNA4+"
    comptime assert bit_width_of[dtype]() == 32, "Unsupported dtype"
    comptime assert stride in (16, 32), "Unsupported stride"

    comptime asm = "llvm.amdgcn.permlane" + String(stride) + ".swap"
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
](val: SIMD[dtype, simd_width], out res: type_of(val)):
    """Shuffles SIMD values across lanes using AMD permlane operations.

    Parameters:
        dtype: Data type of the values.
        simd_width: Width of the SIMD vector.
        stride: Shuffle stride.

    Args:
        val: Input SIMD vector to shuffle.

    Returns:
        Shuffled SIMD vector in the `res` output parameter.
    """
    var lane_group = lane_id() // UInt(stride)

    var out = type_of(res)()

    comptime for i in range(simd_width):
        out[i] = permlane_swap[stride](val[i], val[i])[
            Int((lane_group + 1) % 2)
        ]
    return out
