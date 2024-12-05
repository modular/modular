# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from sys._assembly import inlined_assembly
from sys.info import alignof, bitwidthof, _current_arch
from sys.intrinsics import llvm_intrinsic

from builtin.dtype import _int_type_of_width
from memory import UnsafePointer
from memory.unsafe import bitcast

from .memory import AddressSpace, _int_to_str
from .host.info import Info, H100

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


@always_inline("nodebug")
fn _get_sm_name() -> StringLiteral:
    return _current_arch()


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
    if Info.from_name[_get_sm_name()]() >= H100:
        inlined_assembly[
            "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32",
            NoneType,
            constraints="r",
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
    if Info.from_name[_get_sm_name()]() >= H100:
        inlined_assembly[
            "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
            NoneType,
            constraints="r",
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

    fn _mnemonic(self) -> StringLiteral:
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
    alias suffix = "gl" if scope is Scope.GPU else scope._mnemonic()
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
    alias constraints = _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    alias scope_str = scope._mnemonic()
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
    alias constraints = "=" + _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    alias scope_str = scope._mnemonic()
    return inlined_assembly[
        "ld.acquire."
        + ((scope_str + ".") if scope_str else "")
        + "global."
        + _get_type_suffix[type]()
        + " $0, [$1];",
        Scalar[type],
        constraints=constraints,
    ](ptr.bitcast[address_space = AddressSpace.GENERIC]())


@always_inline
fn store_volatile[
    type: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_], value: Scalar[type]):
    alias constraints = _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    inlined_assembly[
        "st.volatile.global." + _get_type_suffix[type]() + " [$1], $0;",
        NoneType,
        constraints=constraints,
    ](value, ptr.bitcast[address_space = AddressSpace.GENERIC]())


@always_inline
fn load_volatile[
    type: DType, //, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], **_]) -> Scalar[type]:
    alias constraints = "=" + _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    return inlined_assembly[
        "ld.volatile.global." + _get_type_suffix[type]() + " $0, [$1];",
        Scalar[type],
        constraints=constraints,
    ](ptr.bitcast[address_space = AddressSpace.GENERIC]())
