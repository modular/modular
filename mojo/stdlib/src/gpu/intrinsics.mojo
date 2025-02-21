# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

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
        Scalar[type]: The loaded register variable.
    """
    constrained[type.is_numeric(), "the type must be numeric"]()
    return x.load[width=width, alignment=alignment, invariant=True]()


# ===-----------------------------------------------------------------------===#
# warpgroup_reg
# ===-----------------------------------------------------------------------===#


fn warpgroup_reg_alloc[count: Int]():
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to request additional registers such that the absolute per-thread
    maximum register count is increased from its current value to imm-reg-count.
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
    """Provides a hint to the system to update the maximum number of per-thread
    registers owned by the executing warp to the value specified by the
    imm-reg-count operand.

    Used to release extra registers such that the absolute per-thread maximum
    register count is reduced from its current value to imm-reg-count.
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
    """Performs arbitrary logical operation on 3 inputs."""

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
    """Return selected bytes from two 32-bit unsigned integers."""
    alias asm = "llvm.nvvm.prmt" if is_nvidia_gpu() else "llvm.amdgcn.perm"
    return llvm_intrinsic[asm, UInt32, has_side_effect=False](a, b, c)


# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@always_inline
fn mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculate the most significant 32 bits of the product of the two UInt16s.
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
    """Calculate the most significant 32 bits of the product of the two Int16s.
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
    """Calculate the most significant 32 bits of the product of the two UInts.
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
    """Calculate the most significant 32 bits of the product of the two Ints."""

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
    """Calculate the most significant 32 bits of the product of the two UInts.
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
    """Calculate the most significant 32 bits of the product of the two Ints."""

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
    var _value: Int

    alias NONE = Self(0)
    alias THREAD = Self(1)
    alias WARP = Self(2)
    alias BLOCK = Self(3)
    alias CLUSTER = Self(4)
    alias GPU = Self(5)
    alias SYSTEM = Self(6)

    fn __eq__(self, other: Self) -> Bool:
        return self is other

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __isnot__(self, other: Self) -> Bool:
        return not (self is other)

    fn mnemonic(self) -> StringLiteral:
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
    """Memory fence functions can be used to enforce some ordering on memory
    accesses."""
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
    constrained[
        is_nvidia_gpu(), "load_volatile is not currently supported on AMD GPUs"
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
    constrained[
        is_nvidia_gpu(), "load_volatile is not currently supported on AMD GPUs"
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
    constrained[
        is_nvidia_gpu(), "load_volatile is not currently supported on AMD GPUs"
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


# ===-----------------------------------------------------------------------===#
# buffer_load_lds
# ===-----------------------------------------------------------------------===#


@always_inline
fn _make_buffer_resource[
    type: DType
](gds_ptr: UnsafePointer[Scalar[type]], elements: Int) -> SIMD[DType.uint32, 4]:
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
    resource_constant[2] = sizeof[type]() * elements
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
    rsrc: SIMD[DType.uint32, 4],
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
    gds_ptr: UnsafePointer[Scalar[type]],
    gds_offset: Int32,
    lds_ptr_base: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED
    ],
    lds_offset: Int32,
    num_records: Int,
):
    """Loads four bytes from global memory ands writes them to shared memory.

    This function is used to copy from global memory to shared memory (aka LDS)
    bypassing storing to register without waiting for the copy to finish.
    A call to wait_cnt_amd() is necessary to ensure the copy is finished.

    Parameters:
        type: The type of the data to be loaded.

    Arg:
        gds_ptr: Global memory base address.
        gds_offset: Global memory offset.
        lds_ptr_base: LDS base address.
        lds_offset: LDS offset.
        num_records: Reads with gds_offset > num_records return 0.

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

    var resource_constant = _make_buffer_resource(gds_ptr, num_records)

    var global_offset_bytes = Scalar[DType.int32](sizeof[type]() * gds_offset)
    inlined_assembly[
        "buffer_load_dword $0, $1, 0 offen lds",
        NoneType,
        constraints="v,s,~{memory}",
        has_side_effect=True,
    ](global_offset_bytes, resource_constant)


@always_inline
fn buffer_load_store_lds[
    type: DType
](
    gds_ptr: UnsafePointer[Scalar[type]],
    gds_offset: Int32,
    lds_ptr_base: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED
    ],
    lds_offset: Int32,
    num_records: Int,
):
    """Loads four bytes from global memory ands writes them to shared memory.

    This function is used to copy from global memory to shared memory (aka LDS)
    bypassing storing to register.

    Parameters:
        type: The type of the data to be loaded.

    Args:
        gds_ptr: Global memory base address.
        gds_offset: Global memory offset.
        lds_ptr_base: LDS base address.
        lds_offset: LDS offset.
        num_records: Reads with gds_offset > num_records return 0.

    """
    constrained[
        is_amd_gpu(),
        (
            "The buffer_load_store_lds  function is only applicable on AMDGPU"
            " hardware."
        ),
    ]()

    var lds_ptr = lds_ptr_base + lds_offset
    var src_resource = _make_buffer_resource(gds_ptr, num_records)
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
](
    gds_ptr: UnsafePointer[Scalar[type]],
    gds_offset: Int32,
    num_records: Int,
) -> SIMD[type, width]:
    """Loads a register variable from global memory.

    This function is used to copy from global memory to register.

    Parameters:
        type: The type of the data to be loaded.
        width: The width of the SIMD vector to load.

    Args:
        gds_ptr: Global memory base address.
        gds_offset: Global memory offset.
        num_records: Reads with gds_offset > num_records return 0.

    """

    constrained[
        is_amd_gpu(),
        "The buffer_load function is only applicable on AMDGPU hardware.",
    ]()

    alias bytes = sizeof[type]() * width
    var src_resource = _make_buffer_resource(gds_ptr, num_records)
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
](
    gds_ptr: UnsafePointer[Scalar[type]],
    gds_offset: Int32,
    num_records: Int,
    val: SIMD[type, width],
):
    """Stores  a register variable to global memory.

    This function is used to write to global memory from a register.

    Parameters:
        type: The type of the data to be loaded.
        width: The width of the SIMD vector to load.

    Args:
        gds_ptr: Global memory base address.
        gds_offset: Global memory offset.
        num_records: Reads with gds_offset > num_records return 0.
        val: The value to write to memory.
    """

    constrained[
        is_amd_gpu(),
        "The buffer_store function is only applicable on AMDGPU hardware.",
    ]()

    alias bytes = sizeof[type]() * width

    var src_resource = _make_buffer_resource(gds_ptr, num_records)
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
