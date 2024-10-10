# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs intrinsics operations."""

from sys._assembly import inlined_assembly
from sys.info import alignof, bitwidthof
from sys.intrinsics import llvm_intrinsic

from builtin.dtype import _int_type_of_width
from memory import UnsafePointer
from memory.unsafe import bitcast

from .sys import is_sm_greater_equal
from .memory import _int_to_str, AddressSpace

# ===----------------------------------------------------------------------===#
# ldg
# ===----------------------------------------------------------------------===#


@always_inline
fn _bitwidthof_str[type: DType]() -> StringLiteral:
    alias bitwidth = bitwidthof[type]()

    @parameter
    if bitwidth == 8:
        return "8"
    elif bitwidth == 16:
        return "16"
    elif bitwidth == 32:
        return "32"
    elif bitwidth == 64:
        return "64"
    constrained[False, "invalid dtype"]()
    return "invalid"


@always_inline
fn ldg[type: DType](x: UnsafePointer[Scalar[type]]) -> Scalar[type]:
    """Load a register variable from global state space via non-coherent cache.
    """
    constrained[type.is_numeric(), "the type must be numeric"]()

    alias prefix = "llvm.nvvm.ldg.global."
    alias suffix = _bitwidthof_str[type]()

    alias alignment = Int32(alignof[type]())

    @parameter
    if type.is_integral():
        alias integral_type = _int_type_of_width[bitwidthof[type]()]()
        return bitcast[type, 1](
            llvm_intrinsic[prefix + "i.i" + suffix, Scalar[integral_type]](
                x.bitcast[integral_type](), alignment
            )
        )

    constrained[type.is_floating_point(), "the type must be floating point"]()
    return llvm_intrinsic[prefix + "f.f" + suffix, Scalar[type]](x, alignment)


# ===----------------------------------------------------------------------===#
# warpgroup_reg
# ===----------------------------------------------------------------------===#


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
    if is_sm_greater_equal["sm_90a"]():
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
    if is_sm_greater_equal["sm_90a"]():
        inlined_assembly[
            "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
            NoneType,
            constraints="r",
        ](Int32(count))


# ===----------------------------------------------------------------------===#
# clock
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn clock() -> UInt:
    """Returns a 32-bit unsigned cycle counter."""
    return int(llvm_intrinsic["llvm.nvvm.read.ptx.sreg.clock", Int32]())


@always_inline("nodebug")
fn clock64() -> UInt:
    """Returns a 64-bit unsigned cycle counter."""
    return int(llvm_intrinsic["llvm.nvvm.read.ptx.sreg.clock64", Int64]())


# ===----------------------------------------------------------------------===#
# lop
# ===----------------------------------------------------------------------===#


@always_inline
fn lop[lut: Int32](a: Int32, b: Int32, c: Int32) -> Int32:
    """Performs arbitrary logical operation on 3 inputs."""

    return inlined_assembly[
        "lop3.b32 $0, $1, $2, $3, $4;",
        Int32,
        constraints="=r,r,n,n,n",
        has_side_effect=False,
    ](a, b, c, Int32(lut))


# ===----------------------------------------------------------------------===#
# permute
# ===----------------------------------------------------------------------===#


@always_inline
fn byte_permute(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
    """Return selected bytes from two 32-bit unsigned integers."""

    return llvm_intrinsic["llvm.nvvm.prmt", UInt32, has_side_effect=False](
        a, b, c
    )


# ===----------------------------------------------------------------------===#
# mul24
# ===----------------------------------------------------------------------===#


@always_inline
fn mul24(a: Int, b: Int) -> Int:
    """Calculate the least significant 32 bits of the product of the least
    significant 24 bits of two integers.
    """
    return int(mul24(Int32(a), Int32(b)))


@always_inline
fn mul24(a: UInt, b: UInt) -> UInt:
    """Calculate the least significant 32 bits of the product of the least
    significant 24 bits of two integers.
    """
    return int(mul24(UInt32(a), UInt32(b)))


@always_inline
fn mul24(a: UInt32, b: UInt32) -> UInt32:
    """Calculate the least significant 32 bits of the product of the least
    significant 24 bits of two integers.
    """
    return llvm_intrinsic["llvm.nvvm.mullo.ui", UInt32, has_side_effect=False](
        a, b
    )


@always_inline
fn mul24(a: Int32, b: Int32) -> Int32:
    """Calculate the least significant 32 bits of the product of the least
    significant 24 bits of two integers.
    """
    return llvm_intrinsic["llvm.nvvm.mullo.i", Int32, has_side_effect=False](
        a, b
    )


# ===----------------------------------------------------------------------===#
# mulhi
# ===----------------------------------------------------------------------===#


@always_inline
fn mulhi(a: Int, b: Int) -> Int:
    """Calculate the most significant 32 bits of the product of the two integers.
    """
    return int(mulhi(Int32(a), Int32(b)))


@always_inline
fn mulhi(a: UInt, b: UInt) -> UInt:
    """Calculate the most significant 32 bits of the product of the two UInts.
    """
    return int(mulhi(UInt32(a), UInt32(b)))


@always_inline
fn mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculate the most significant 32 bits of the product of the two UInts.
    """
    return llvm_intrinsic["llvm.nvvm.mulhi.us", UInt32, has_side_effect=False](
        a, b
    )


@always_inline
fn mulhi(a: Int16, b: Int16) -> Int32:
    """Calculate the most significant 32 bits of the product of the two Ints."""
    return llvm_intrinsic["llvm.nvvm.mulhi.s", Int32, has_side_effect=False](
        a, b
    )


@always_inline
fn mulhi(a: UInt32, b: UInt32) -> UInt32:
    """Calculate the most significant 32 bits of the product of the two UInts.
    """
    return llvm_intrinsic["llvm.nvvm.mulhi.ui", UInt32, has_side_effect=False](
        a, b
    )


@always_inline
fn mulhi(a: Int32, b: Int32) -> Int32:
    """Calculate the most significant 32 bits of the product of the two Ints."""
    return llvm_intrinsic["llvm.nvvm.mulhi.i", Int32, has_side_effect=False](
        a, b
    )


@always_inline
fn mulhi(a: UInt64, b: UInt64) -> UInt64:
    """Calculate the most significant 32 bits of the product of the two Ints."""
    return llvm_intrinsic["llvm.nvvm.mulhi.ull", UInt64, has_side_effect=False](
        a, b
    )


@always_inline
fn mulhi(a: Int64, b: Int64) -> Int64:
    """Calculate the most significant 32 bits of the product of the two Ints."""
    return llvm_intrinsic["llvm.nvvm.mulhi.ll", Int64, has_side_effect=False](
        a, b
    )


# ===----------------------------------------------------------------------===#
# mulwide
# ===----------------------------------------------------------------------===#


@always_inline
fn mulwide(a: UInt32, b: UInt32) -> UInt64:
    """Calculate the most significant 32 bits of the product of the two UInts.
    """
    return inlined_assembly[
        "mul.wide.u32 $0, $1, $2;",
        UInt64,
        constraints="=l,r,r",
        has_side_effect=False,
    ](a, b)


@always_inline
fn mulwide(a: Int32, b: Int32) -> Int64:
    """Calculate the most significant 32 bits of the product of the two Ints."""
    return inlined_assembly[
        "mul.wide.s32 $0, $1, $2;",
        Int64,
        constraints="=l,r,r",
        has_side_effect=False,
    ](a, b)


# ===----------------------------------------------------------------------===#
# threadfence
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# release / acquire
# ===----------------------------------------------------------------------===#


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
](ptr: UnsafePointer[Scalar[type], *_, **_], value: Scalar[type]):
    alias constraints = _get_pointer_constraint() + "," + _get_register_constraint[
        type
    ]() + (
        ",~{memory}" if memory else ""
    )
    alias scope_str = scope._mnemonic()
    inlined_assembly[
        "st.release."
        + scope_str
        + ".global."
        + _get_type_suffix[type]()
        + " [$0], $1;",
        NoneType,
        constraints=constraints,
    ](ptr.bitcast[address_space = AddressSpace.GENERIC](), value)


@always_inline
fn load_acquire[
    type: DType, //, *, scope: Scope = Scope.SYSTEM, memory: Bool = True
](ptr: UnsafePointer[Scalar[type], *_, **_]) -> Scalar[type]:
    alias constraints = "=" + _get_register_constraint[
        type
    ]() + "," + _get_pointer_constraint() + (",~{memory}" if memory else "")
    alias scope_str = scope._mnemonic()
    return inlined_assembly[
        "ld.acquire."
        + scope_str
        + ".global."
        + _get_type_suffix[type]()
        + " $0, [$1];",
        Scalar[type],
        constraints=constraints,
    ](ptr.bitcast[address_space = AddressSpace.GENERIC]())
