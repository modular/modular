# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Contains wrappers of functions that are (or used to be) in the `math` module.

This is needed because MOGG expects these to have a certain signature, but that
shouldn't restrict these functions and what the `math` module contains.
"""

import math

from register import mogg_register


# ===----------------------------------------------------------------------===#
# Basic elementwise primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mo.mod")
@always_inline
fn mod[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise modulo operation of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: The numerator of the operation.
        y: The denominator of the operation.

    Returns:
        Elementwise remainder of x divided by y.
    """
    return x % y


@mogg_register("mo.mul")
@always_inline
fn mul[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise multiplication of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: First SIMD vector to multiply.
        y: Second SIMD vector to multiply.

    Returns:
        Elementwise multiplication of x and y.
    """
    return x * y


@mogg_register("mo.sub")
@always_inline
fn sub[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise subtraction of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: SIMD vector which y will be subtracted from.
        y: SIMD vector to subtract from x.

    Returns:
        Elementwise subtraction of x and y.
    """
    return x - y


@mogg_register("mo.add")
@always_inline
fn add[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise addition of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: First SIMD vector to add.
        y: Second SIMD vector to add.

    Returns:
        Elementwise addition of x and y.
    """
    return x + y


@mogg_register("mo.div")
@always_inline
fn div[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise division of two SIMD vectors.

    Parameters:
        type: DType of the input SIMD vectors.
        simd_width: Width of the input SIMD vectors.

    Args:
        x: SIMD vector containing the dividends.
        y: SIMD vector containing the quotients.

    Returns:
        Elementwise division of SIMD vector x by SIMD vector y (this is x / y).
    """
    return x / y


# ===----------------------------------------------------------------------=== #
# ceil
# ===----------------------------------------------------------------------=== #


@mogg_register("mo.ceil")
@always_inline
fn ceil[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Ceil Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Ceil operation on.

    Returns:
        The result of the Ceil operation.
    """
    return math.ceil(x)


# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


@mogg_register("mo.floor")
@always_inline
fn floor[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Floor Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Floor operation on.

    Returns:
        The result of the Floor operation.
    """
    return math.floor(x)


# ===----------------------------------------------------------------------=== #
# tanh
# ===----------------------------------------------------------------------=== #


@mogg_register("mo.tanh")
@always_inline
fn tanh[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Tanh Op.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the Tanh operation on.

    Returns:
        The result of the Tanh operation.
    """
    return math.tanh(x)
