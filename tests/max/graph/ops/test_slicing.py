# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import random
from functools import reduce
from typing import Optional

from conftest import broadcast_shapes, broadcastable_tensor_types, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Dim, Graph, StaticDim, TensorType, ops


@given(input_types=broadcastable_tensor_types(3))
def test_select(input_types: list[TensorType]):
    input_types[0].dtype = DType.bool

    with Graph("select", input_types=input_types) as graph:
        cond, x, y = graph.inputs
        out = ops.select(cond, x, y)

        expected = reduce(broadcast_shapes, (t.shape for t in input_types))
        assert out.shape == expected
        assert out.dtype in (t.dtype for t in input_types)

        graph.output(out)


def test_slice_basic():
    with Graph(
        "slice", input_types=[TensorType(DType.int32, [1, 2, 3, 4, 5])]
    ) as graph:
        out = graph.inputs[0][:, 1, ..., 3]

        assert out.shape == [1, 3, 4]
        graph.output(out)


def test_slice_with_tensor_value():
    with Graph(
        "slice", input_types=[TensorType(DType.int32, [5, "in_dim"])]
    ) as graph:
        start = ops.constant(2, DType.int64)
        out = graph.inputs[0][
            (slice(start, None), 3), (slice(start, None), "out_dim")
        ]

        assert out.shape == [3, "out_dim"]
        graph.output(out)


def dim_indexes(dim: Dim):
    assume(dim != 0)  # still need to test attempting to index with 0 dim
    # Can index symbolic dims at any index, checked at runtime.
    bound = dim.dim if isinstance(dim, StaticDim) else (2**63 - 1)
    return st.one_of(
        # `:` include whole dim.
        st.just(slice(None, None, None)),
        st.integers(-bound, bound - 1),
    )


def shape_indexes(shape: list[Dim]):
    full_indexes = st.tuples(*(dim_indexes(dim) for dim in shape))

    def with_ellipsis(index, slice):
        # Ellipses can only be contiguous indices.
        assume(slice.step in (None, 1))
        new_index = list(index)
        new_index[slice] = [...]
        return new_index

    indexes_with_ellipsis = full_indexes.flatmap(
        lambda index: st.slices(len(shape)).map(
            lambda slice: with_ellipsis(index, slice)
        )
    )

    return full_indexes | indexes_with_ellipsis


# can remove 0 from possible dims here
shared_shapes = st.shared(st.from_type(list[Dim]))


def expected_slice_shape(shape, index):
    if Ellipsis in index:
        # Split around Ellipsis, fill its with slice(None)
        ei = index.index(Ellipsis)
        left, right = index[:ei], index[ei + 1 :]
        elen = len(shape) - (len(index) - 1)
        effective_index = [*left, *([slice(None)] * elen), *right]
    else:
        effective_index = index

    assert len(effective_index) == len(shape)

    def expected_dim(dim, dim_index):
        if dim_index == slice(None):
            return dim
        elif isinstance(dim_index, int):
            return None
        elif isinstance(dim_index, slice):
            return len(range(*dim_index.indices(dim.dim)))
        # support more slicing cases
        raise NotImplementedError

    expected = (
        expected_dim(dim, idx) for dim, idx in zip(shape, effective_index)
    )
    return [dim for dim in expected if dim is not None]


@given(
    tensor_type=tensor_types(shapes=shared_shapes),
    index=shared_shapes.flatmap(shape_indexes),
)
def test_slice_valid_ints(tensor_type: TensorType, index):
    assume(tensor_type.shape)
    assume(0 not in tensor_type.shape)

    with Graph("slice", input_types=[tensor_type]) as graph:
        out = ops.slice_tensor(graph.inputs[0], index)
        assert out.shape == expected_slice_shape(tensor_type.shape, index)
        graph.output(out)


def gen_slice(n, rand: random.Random):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    if rand.randint(0, 1):
        start = rand.randint(-1 * n, n - 1)
    if rand.randint(0, 1):
        step = rand.randint(-1 * n, n) or 1
    if rand.randint(0, 1):
        stop = rand.randint(-1 * n, n)

    return slice(start, stop, step)


static_tensor_type = tensor_types(
    shapes=st.shared(st.from_type(list[StaticDim]))
)


@given(tensor_type=static_tensor_type, rand=...)
def test_slice_static_dims(tensor_type: TensorType, rand: random.Random):
    assume(tensor_type.shape)
    assume(0 not in tensor_type.shape)

    index = [gen_slice(d.dim, rand) for d in tensor_type.shape]

    with Graph("slice", input_types=[tensor_type]) as graph:
        out = ops.slice_tensor(graph.inputs[0], index)
        assert out.shape == expected_slice_shape(tensor_type.shape, index)
        graph.output(out)
