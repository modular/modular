# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides low-level GPU intrinsic operations and memory access primitives.

The module implements hardware-specific intrinsic operations that map directly to GPU
assembly instructions, with a focus on NVIDIA GPU architectures. It includes:

- Global memory load/store operations with cache control
- Warp-level primitives and synchronization
- Memory fence and barrier operations
- Atomic operations and memory ordering primitives

These low-level primitives should be used carefully as they correspond directly to
hardware instructions and require understanding of the underlying GPU architecture.
"""

from sys._assembly import inlined_assembly
from sys.info import _current_arch, alignof, bitwidthof, _is_sm_9x
from sys.intrinsics import llvm_intrinsic

from builtin.dtype import _int_type_of_width
from memory import UnsafePointer
from memory.unsafe import bitcast

from .host.info import H100, Info
from .memory import AddressSpace, _int_to_str

from sys.intrinsics import readfirstlane


# ===-----------------------------------------------------------------------===#
# ldg
# ===-----------------------------------------------------------------------===#


@always_inline
fn ldg[
    type: DType, //,
    width: Int = 1,
    *,
    alignment: Int = alignof[SIMD[type, width]](),
](x: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    """Load a register variable from global state space via non-coherent cache.

    This function is used to load a register variable from the global state
    space through a non-coherent cache. The type of the data to be loaded must
    be numeric.

    Parameters:
        type: The type of the data to be loaded.
        width: The width of the SIMD vector to load.
        alignment: The alignment of the data in bytes. Defaults to the
            alignment of the type.

    Returns:
        The loaded register variable.
    """
    constrained[type.is_numeric(), "the type must be numeric"]()
    return x.load[width=width, alignment=alignment, invariant=True]()


# ===-----------------------------------------------------------------------===#
# warpgroup_reg
# ===-----------------------------------------------------------------------===#


fn warpgroup_reg_alloc[count: Int]():
    """Allocates additional registers for the executing warp group.

    This function provides a hint to the system to increase the maximum number of
    per-thread registers owned by the executing warp. It requests additional registers
    to increase the absolute per-thread maximum register count from its current value
    to the specified count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs
        - This is a performance optimization hint that may be ignored by the hardware
        - Should be paired with `warpgroup_reg_dealloc() when extra registers are no
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

    This function provides a hint to the system to decrease the maximum number of
    per-thread registers owned by the executing warp. It releases extra registers
    to reduce the absolute per-thread maximum register count from its current value
    to the specified count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8.
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - This is a performance optimization hint that may be ignored by the hardware.
        - Should be paired with `warpgroup_reg_alloc()` when extra registers are needed.
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
    """Performs an arbitrary logical operation on 3 input values using a lookup table.

    This function implements a 3-input lookup table (LUT) operation, where the result is
    determined by the bits in the lookup table value for each possible input combination.

    Parameters:
        lut: A 32-bit lookup table value that defines the logical operation.

    Args:
        a: First input value.
        b: Second input value.
        c: Third input value.

    Returns:
        The result of applying the lookup table operation to the inputs.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps to the LOP3.B32 PTX instruction.
        - The lookup table value determines the output for each possible input combination.
    """

    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "lop3.b32 $0, $1, $2, $3, $4;",
            Int32,
            constraints="=r,r,n,n,n",
            has_side_effect=False,
        ](a, b, c, Int32(lut))

    constrained[False, "The lop function is not supported by AMD GPUs."]()
    return abort[Int32]("function not available")


# ===-----------------------------------------------------------------------===#
# permute
# ===-----------------------------------------------------------------------===#


@always_inline
fn byte_permute(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
    """Permutes bytes from two 32-bit integers based on a control mask.

    This function selects and rearranges individual bytes from two source integers
    based on a control mask to create a new 32-bit value.

    Args:
        a: First source integer containing bytes to select from.
        b: Second source integer containing bytes to select from.
        c: Control mask that specifies which bytes to select and their positions
           Each byte in the mask controls the selection and placement of one output byte.

    Returns:
        A new 32-bit integer containing the selected and rearranged bytes

    Note:
        The exact byte selection behavior depends on the GPU architecture:
        - On NVIDIA: Maps to PRMT instruction
        - On AMD: Maps to PERM instruction.
    """
    """Return selected bytes from two 32-bit unsigned integers."""
    alias asm = "llvm.nvvm.prmt" if is_nvidia_gpu() else "llvm.amdgcn.perm"
    return llvm_intrinsic[asm, UInt32, has_side_effect=False](a, b, c)


# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@always_inline
fn mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 16-bit unsigned integers.

    This function multiplies two 16-bit unsigned integers and returns the high 32 bits
    of their product. This is useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 16-bit unsigned integer operand.
        b: Second 16-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U16 PTX instruction.
        On other architectures, it performs the multiplication using 32-bit arithmetic.
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
    """Calculates the most significant 32 bits of the product of two 16-bit signed integers.

    This function multiplies two 16-bit signed integers and returns the high 32 bits
    of their product. This is useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 16-bit signed integer operand.
        b: Second 16-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S16 PTX instruction.
        On other architectures, it performs the multiplication using 32-bit arithmetic.
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
    """Calculates the most significant 32 bits of the product of two 32-bit unsigned integers.

    This function multiplies two 32-bit unsigned integers and returns the high 32 bits
    of their product. This is useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U32 PTX instruction.
        On other architectures, it performs the multiplication using 64-bit arithmetic.
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
    """Calculates the most significant 32 bits of the product of two 32-bit signed integers.

    This function multiplies two 32-bit signed integers and returns the high 32 bits
    of their product. This is useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S32 PTX instruction.
        On other architectures, it performs the multiplication using 64-bit arithmetic.
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
    This is useful when the product may exceed 32 bits.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The full 64-bit product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.U32 PTX instruction.
        On other architectures, it performs the multiplication using 64-bit casts.
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
    This is useful when the product may exceed 32 bits or be negative.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The full 64-bit signed product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.S32 PTX instruction.
        On other architectures, it performs the multiplication using 64-bit casts.
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


# ===-----------------------------------------------------------------------===#
# threadfence
# ===-----------------------------------------------------------------------===#


@value
struct Scope:
    """Represents memory synchronization scope levels for GPU memory operations.

    This struct defines different scopes of memory visibility and synchronization,
    from thread-local to system-wide. Each scope level determines how memory
    operations are ordered and visible across different execution units.

    The scope levels form a hierarchy, with each higher level providing stronger
    ordering guarantees but potentially higher synchronization costs.
    """

    var _value: Int

    alias NONE = Self(0)
    """No memory ordering guarantees. Operations may be reordered freely."""

    alias THREAD = Self(1)
    """Thread-level scope. Memory operations are ordered within a single thread."""

    alias WARP = Self(2)
    """Warp-level scope. Memory operations are ordered within a warp of threads."""

    alias BLOCK = Self(3)
    """Block-level scope. Memory operations are ordered within a thread block/CTA."""

    alias CLUSTER = Self(4)
    """Cluster-level scope. Memory operations are ordered within a thread block cluster."""

    alias GPU = Self(5)
    """GPU-level scope. Memory operations are ordered across all threads on the GPU."""

    alias SYSTEM = Self(6)
    """System-wide scope. Memory operations are ordered across the entire system."""

    fn __eq__(self, other: Self) -> Bool:
        return self is other

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __isnot__(self, other: Self) -> Bool:
        return not (self is other)

    fn mnemonic(self) -> StringLiteral:
        """Returns the mnemonic string representation of the memory scope.

        This method converts the memory scope level into a string mnemonic used by
        LLVM/NVVM intrinsics for memory operations.

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

    This function acts as a memory fence/barrier that ensures all memory operations
    (both loads and stores) issued before the fence are visible to other threads
    within the specified scope before any memory operations after the fence.

    Parameters:
        scope: The memory scope level for the fence operation. Defaults to GPU-wide scope.
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
    alias suffix = "gl" if scope is Scope.GPU else scope.mnemonic()
    llvm_intrinsic["llvm.nvvm.membar." + suffix, NoneType]()


# ===-----------------------------------------------------------------------===#
# release / acquire
# ===-----------------------------------------------------------------------===#


fn _get_type_suffix[type: DType]() -> StringLiteral:
    alias str = "u" + _int_to_str[bitwidthof[type]()]()
    return str


fn _get_register_constraint[type: DType]() -> StringLiteral:
    if type is DType.bool:
        return "b"
    if type.is_half_float():
        return "h"
    if type.is_integral():
        alias width = bitwidthof[type]()
        if width == 16:
            return "c"
        if width == 32:
            return "r"
        if width == 64:
            return "l"
    if type is DType.float32:
        return "f"
    if type is DType.float64:
        return "d"

    return "<<unknown_register_constraint>>"


fn _get_pointer_constraint() -> StringLiteral:
    return _get_register_constraint[DType.index]()


@always_inline
fn store_release[
    type: DType, //, scope: Scope = Scope.SYSTEM, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_], value: Scalar[type]):
    """Performs an atomic store operation with release memory ordering semantics.

    Args:
        ptr: Pointer to the memory location to store to.
        value: Value to store.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps directly to PTX st.release instruction.
        - Ensures all previous memory operations complete before this store.
    """
    constrained[
        is_nvidia_gpu(), "store_release is not currently supported on AMD GPUs"
    ]()
    alias constraints = _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    alias scope_str = scope.mnemonic()
    inlined_assembly[
        "st.release."
        + ((scope_str + ".") if scope_str else "")
        + "global."
        + _get_type_suffix[type]()
        + " [$1], $0;",
        NoneType,
        constraints=constraints,
    ](value, ptr)


@always_inline
fn load_acquire[
    type: DType, //, *, scope: Scope = Scope.SYSTEM, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_]) -> Scalar[type]:
    """Performs an atomic load operation with acquire memory ordering semantics.

    Args:
        ptr: Pointer to the memory location to load from.

    Returns:
        The loaded value

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps directly to PTX ld.acquire instruction.
        - Ensures subsequent memory operations don't execute until after this load.
    """
    constrained[
        is_nvidia_gpu(), "load_acquire is not currently supported on AMD GPUs"
    ]()
    alias constraints = "=" + _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    alias scope_str = scope.mnemonic()
    return inlined_assembly[
        "ld.acquire."
        + ((scope_str + ".") if scope_str else "")
        + "global."
        + _get_type_suffix[type]()
        + " $0, [$1];",
        Scalar[type],
        constraints=constraints,
    ](ptr.address_space_cast[AddressSpace.GENERIC]())


@always_inline
fn store_volatile[
    type: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_], value: Scalar[type]):
    """Performs a volatile store operation that cannot be optimized away.

    Args:
        ptr: Pointer to the memory location to store to.
        value: Value to store.

    Note:
        - Only supported on NVIDIA GPUs
        - Maps directly to PTX st.volatile instruction
        - Prevents compiler optimization of the store operation
        - Useful for memory-mapped I/O or synchronization primitives
    """
    constrained[
        is_nvidia_gpu(), "store_volatile is not currently supported on AMD GPUs"
    ]()
    alias constraints = _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    inlined_assembly[
        "st.volatile.global." + _get_type_suffix[type]() + " [$1], $0;",
        NoneType,
        constraints=constraints,
    ](value, ptr.address_space_cast[AddressSpace.GENERIC]())


@always_inline
fn load_volatile[
    type: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_]) -> Scalar[type]:
    """Performs a volatile load operation that cannot be optimized away.

    Args:
        ptr: Pointer to the memory location to load from.

    Returns:
        The loaded value.

    Note:
        - Only supported on NVIDIA GPUs
        - Maps directly to PTX ld.volatile instruction
        - Prevents compiler optimization of the load operation
        - Useful for memory-mapped I/O or synchronization primitives.
    """
    constrained[
        is_nvidia_gpu(), "load_volatile is not currently supported on AMD GPUs"
    ]()
    alias constraints = "=" + _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    return inlined_assembly[
        "ld.volatile.global." + _get_type_suffix[type]() + " $0, [$1];",
        Scalar[type],
        constraints=constraints,
    ](ptr.address_space_cast[AddressSpace.GENERIC]())


alias _buffer_resource = SIMD[DType.uint32, 4]
"""A 128-bit descriptor used to describe a buffer resource on AMD GPUs. Used for buffer_load/buffer_store instructions."""


@always_inline
fn make_buffer_resource[
    type: DType
](
    gds_ptr: UnsafePointer[Scalar[type], **_],
    num_records: Int = Int(UInt32.MAX),
) -> _buffer_resource:
    """Creates a 128-bit buffer constant for buffer IO.

    This function is used to create a 128-bit buffer constant struct for buffer IO

    Parameters:
        type: The type of the data to be loaded.

    Arg:
        gds_ptr: Global memory base address.
        num_records: Reads with gds_offset > num_records return 0. Defaults to the maximum number of elements UInt32.MAX.

    """

    constrained[
        is_amd_gpu(),
        (
            "The make_buffer_resource function is only applicable on AMDGPU"
            " hardware."
        ),
    ]()

    # https://discourse.llvm.org/t/representing-buffer-descriptors-in-the-amdgpu-target-call-for-suggestions/68798/1
    # llvm.amdgcn.make.buffer.rsrc intrinsic, which takes the pointer, which becomes the base of the resource,
    # the 16-bit stride (and swzizzle control) field stored in bits 63:48 of a V#,
    # the 32-bit NumRecords/extent field (bits 95:64),
    # and the 32-bit flags field (bits 127:96).

    var resource_constant = SIMD[DType.uint32, 4](0)
    var address = bitcast[DType.uint32, 2](SIMD[DType.uint64, 1](Int(gds_ptr)))
    resource_constant[0] = address[0]
    # assuming 0 stride currently
    resource_constant[1] = address[1]
    resource_constant[2] = sizeof[type]() * num_records
    # https://github.com/ROCm/composable_kernel/blob/3b2302081eab4975370e29752343058392578bcb/include/ck/ck.hpp#L84
    resource_constant[3] = 0x00020000
    return resource_constant


@always_inline
fn _waitcnt():
    constrained[
        is_amd_gpu(),
        "The _waitcnt function is only applicable on AMDGPU hardware.",
    ]()
    inlined_assembly[
        "s_waitcnt vmcnt(0)",
        NoneType,
        constraints="",
        has_side_effect=True,
    ]()


@always_inline
fn _raw_buffer_load_lds[
    type: DType
](
    rsrc: _buffer_resource,
    lds_ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED],
    size: Int32,
    voffset: Int32,
    soffset: Int32,
    offset: Int32,
    aux: Int32,
):
    constrained[
        is_amd_gpu(),
        (
            "The _raw_buffer_load_lds function is only applicable on AMDGPU"
            " hardware."
        ),
    ]()
    llvm_intrinsic[
        "llvm.amdgcn.raw.buffer.load.lds", NoneType, has_side_effect=True
    ](rsrc, lds_ptr, size, voffset, soffset, offset, aux)


@always_inline
fn _buffer_load_store_lds_nowait[
    type: DType
](
    src_resource: _buffer_resource,
    gds_offset: Int32,
    lds_ptr_base: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED
    ],
    lds_offset: Int32,
):
    """Loads four bytes from global memory ands writes them to shared memory.

    This function is used to copy from global memory to shared memory (aka LDS)
    bypassing storing to register without waiting for the copy to finish.
    A call to wait_cnt_amd() is necessary to ensure the copy is finished.

    Parameters:
        type: The type of the data to be loaded.

    Arg:
        src_resource: The buffer resource descriptor from make_buffer_resource.
        gds_offset: Global memory offset.
        lds_ptr_base: LDS base address.
        lds_offset: LDS offset.
    """

    constrained[
        is_amd_gpu(),
        (
            "The _buffer_load_store_lds_nowait  function is only applicable on"
            " AMDGPU hardware."
        ),
    ]()

    var lds_ptr = lds_ptr_base + lds_offset
    var lds_ptr_sgpr = readfirstlane(SIMD[DType.int32, 1](Int(lds_ptr)))
    inlined_assembly[
        "s_mov_b32 m0, $0",
        NoneType,
        constraints="s,~{memory}",
        has_side_effect=True,
    ](lds_ptr_sgpr)

    var global_offset_bytes = Scalar[DType.int32](sizeof[type]() * gds_offset)
    inlined_assembly[
        "buffer_load_dword $0, $1, 0 offen lds",
        NoneType,
        constraints="v,s,~{memory}",
        has_side_effect=True,
    ](global_offset_bytes, src_resource)


@always_inline
fn buffer_load_store_lds[
    type: DType
](
    src_resource: _buffer_resource,
    gds_offset: Int32,
    lds_ptr_base: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED
    ],
    lds_offset: Int32,
):
    """Loads four bytes from global memory ands writes them to shared memory.

    This function is used to copy from global memory to shared memory (aka LDS)
    bypassing storing to register.

    Parameters:
        type: The type of the data to be loaded.

    Args:
        src_resource: The buffer resource descriptor from make_buffer_resource.
        gds_offset: Global memory offset.
        lds_ptr_base: LDS base address.
        lds_offset: LDS offset.
    """
    constrained[
        is_amd_gpu(),
        (
            "The buffer_load_store_lds  function is only applicable on AMDGPU"
            " hardware."
        ),
    ]()

    var lds_ptr = lds_ptr_base + lds_offset
    var global_offset_bytes = Scalar[DType.int32](sizeof[type]() * gds_offset)
    _raw_buffer_load_lds(
        src_resource,
        lds_ptr,
        sizeof[DType.uint32](),
        global_offset_bytes,
        0,
        0,
        0,
    )


@always_inline
fn buffer_load[
    type: DType, width: Int
](src_resource: _buffer_resource, gds_offset: Int32) -> SIMD[type, width]:
    """Loads a register variable from global memory.

    This function is used to copy from global memory to register.

    Parameters:
        type: The type of the data to be loaded.
        width: The width of the SIMD vector to load.

    Args:
        src_resource: The buffer resource descriptor from make_buffer_resource.
        gds_offset: Global memory offset.
    """

    constrained[
        is_amd_gpu(),
        "The buffer_load function is only applicable on AMDGPU hardware.",
    ]()

    alias bytes = sizeof[type]() * width
    var global_offset_bytes: Int32 = Scalar[DType.int32](
        sizeof[type]() * gds_offset
    )
    # READ
    # GLC = 0 Reads can hit on the L1 and persist across wavefronts
    # GLC = 1 Reads miss the L1 and L2 and force fetch to the data fabric. No L1 persistence across waves.
    alias glc: Int32 = 0
    var src_wave_addr_offset: Int32 = 0

    @parameter
    fn get_inst_name() -> StringLiteral:
        @parameter
        if bytes == 1:
            return "llvm.amdgcn.raw.buffer.load.i8"
        elif bytes == 2:
            return "llvm.amdgcn.raw.buffer.load.i16"
        elif bytes == 4:
            return "llvm.amdgcn.raw.buffer.load.i32"
        elif bytes == 8:
            return "llvm.amdgcn.raw.buffer.load.v2i32"
        elif bytes == 16:
            return "llvm.amdgcn.raw.buffer.load.v4i32"
        else:
            constrained[False, "Width not supported"]()
            return ""

    return llvm_intrinsic[
        get_inst_name(),
        SIMD[type, width],
        has_side_effect=True,
    ](src_resource, global_offset_bytes, src_wave_addr_offset, glc)


@always_inline
fn buffer_store[
    type: DType, width: Int
](src_resource: _buffer_resource, gds_offset: Int32, val: SIMD[type, width],):
    """Stores  a register variable to global memory.

    This function is used to write to global memory from a register.

    Parameters:
        type: The type of the data to be loaded.
        width: The width of the SIMD vector to load.

    Args:
        src_resource: The buffer resource descriptor from make_buffer_resource.
        gds_offset: Global memory offset.
        val: The value to write to memory.
    """

    constrained[
        is_amd_gpu(),
        "The buffer_store function is only applicable on AMDGPU hardware.",
    ]()

    alias bytes = sizeof[type]() * width

    var global_offset_bytes: Int32 = Scalar[DType.int32](
        sizeof[type]() * gds_offset
    )
    # WRITE
    # GLC = 0 Writes miss the L1, write through to L2, and persist in L1 across wavefronts.
    # GLC = 1 Writes miss the L1, write through to L2. No persistence across wavefronts.
    alias glc: Int32 = 0
    var src_wave_addr_offset: Int32 = 0

    @parameter
    fn get_inst_name() -> StringLiteral:
        @parameter
        if bytes == 1:
            return "llvm.amdgcn.raw.buffer.store.i8"
        elif bytes == 2:
            return "llvm.amdgcn.raw.buffer.store.i16"
        elif bytes == 4:
            return "llvm.amdgcn.raw.buffer.store.i32"
        elif bytes == 8:
            return "llvm.amdgcn.raw.buffer.store.v2i32"
        elif bytes == 16:
            return "llvm.amdgcn.raw.buffer.store.v4i32"
        else:
            constrained[False, "Width not supported"]()
            return ""

    llvm_intrinsic[
        get_inst_name(),
        NoneType,
        has_side_effect=True,
    ](val, src_resource, global_offset_bytes, src_wave_addr_offset, glc)
