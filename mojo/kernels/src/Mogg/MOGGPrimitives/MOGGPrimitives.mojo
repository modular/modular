# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import (
    div_ceil,
    max,
    min,
)
from register import *

# ===----------------------------------------------------------------------===#
# MIP Index Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mip.constant.index")
@always_inline
@export
fn mip_constant_index[value: Int]() -> Int:
    return value


@mogg_register("mip.add")
@always_inline
@export
fn mip_add(x: Int, y: Int) -> Int:
    return x + y


@mogg_register("mip.mul")
@always_inline
@export
fn mip_mul(lhs: Int, rhs: Int) -> Int:
    return lhs * rhs


@mogg_register("mip.div")
@always_inline
@export
fn mip_div(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.div divide by zero")
    return numerator // denominator


@mogg_register("mip.div.ceil")
@always_inline
@export
fn mip_div_ceil(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.div.ceil divide by zero")
    return div_ceil(numerator, denominator)


@mogg_register("mip.cmp.eq")
@always_inline
@export
fn mip_cmp_eq(x: Int, y: Int) -> Bool:
    return x == y


@mogg_register("mip.cmp.lt")
@always_inline
@export
fn mip_cmp_lt(x: Int, y: Int) -> Bool:
    return x < y


@mogg_register("mip.cmp.le")
@always_inline
@export
fn mip_cmp_le(x: Int, y: Int) -> Bool:
    return x <= y


@mogg_register("mip.max")
@always_inline
@export
fn mip_max(x: Int, y: Int) -> Int:
    return max(x, y)


@mogg_register("mip.min")
@always_inline
@export
fn mip_min(x: Int, y: Int) -> Int:
    return min(x, y)


@mogg_register("mip.mod")
@always_inline
@export
fn mip_mod(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.mod divide by zero")
    return numerator % denominator


# ===----------------------------------------------------------------------===#
# MIP Bool Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mip.constant.bool")
@always_inline
@export
fn mip_constant_bool[value: Bool]() -> Bool:
    return value


@mogg_register("mip.bool.and")
@always_inline
@export
fn mip_and(x: Bool, y: Bool) -> Bool:
    return x & y


@mogg_register("mip.bool.or")
@always_inline
@export
fn mip_or(x: Bool, y: Bool) -> Bool:
    return x | y


@mogg_register("mip.bool.xor")
@always_inline
@export
fn mip_xor(x: Bool, y: Bool) -> Bool:
    return x ^ y


@mogg_register("mip.select")
@always_inline
@export
fn mip_select[T: AnyRegType](cond: Bool, true: T, false: T) -> T:
    return true if cond else false


@mogg_register("mip.nary.mul")
@always_inline
@export
fn mip_nary_mul[constInt: Int](*vals: Int) -> Int:
    var product = constInt
    for val in vals:
        product *= val
    return product


# ===----------------------------------------------------------------------===#
# MGP Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.assert")
@always_inline
@export
fn mgp_assert[message: StringLiteral](cond: Bool) raises:
    if not cond:
        raise Error(message)
