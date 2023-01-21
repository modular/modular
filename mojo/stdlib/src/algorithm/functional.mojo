# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int
from Assert import assert_param

# ===----------------------------------------------------------------------===#
# Map
# ===----------------------------------------------------------------------===#


@always_inline
fn map[
    func: __mlir_type[`!kgen.signature<(`, Int, `) force_inline -> !lit.none>`],
](size: Int):
    """
    Map a function over a range from 0 to size.
    """
    var i: Int = 0
    while i < size:
        func(i)
        i += 1


# ===----------------------------------------------------------------------===#
# repeat
# ===----------------------------------------------------------------------===#


@always_inline
fn repeat[
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() force_inline -> !lit.none>`],
]():
    """
    Reateadly evaluate a function `count` times.
    """
    _repeat_impl[0, count, func]()


@always_inline
@interface
fn _repeat_impl[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() force_inline -> !lit.none>`],
]():
    ...


@always_inline
@implements(_repeat_impl)
fn _repeat_impl_base[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() force_inline -> !lit.none>`],
]():
    assert_param[idx >= count]()


@always_inline
@implements(_repeat_impl)
fn _repeat_impl_iter[
    idx: __mlir_type.index,
    count: __mlir_type.index,
    func: __mlir_type[`!kgen.signature<<idx>() force_inline -> !lit.none>`],
]():
    assert_param[idx < count]()
    func[idx]()
    _repeat_impl[idx + 1, count, func]()


# ===----------------------------------------------------------------------===#
# Vectorize
# ===----------------------------------------------------------------------===#


@always_inline
fn vectorize[
    simd_width: __mlir_type.index,
    func: __mlir_type[
        `!kgen.signature<<simd_width>(`,
        Int,
        `) force_inline -> !lit.none>`,
    ],
](size: Int):
    """Map a function which is parametrized over a simd_Width over a range
    from 0 to size in simd fashion.
    """
    var i: Int = 0
    let vector_end = (size // simd_width) * simd_width
    while i < vector_end:
        func[simd_width](i)
        i += simd_width
    i = vector_end
    while i < size:
        func[1](i)
        i += 1
