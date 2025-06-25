# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from __future__ import annotations

import operator
import random
from typing import Any

import numpy as np
import pytest
from conftest import tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, StaticDim, TensorType, ops


def test_slice_basic(graph_builder) -> None:
    with graph_builder(
        input_types=[
            TensorType(DType.int32, [1, 2, 3, 4, 5], device=DeviceRef.CPU())
        ],
    ) as graph:
        out = graph.inputs[0][:, 1, ..., 3]

        assert out.shape == [1, 3, 4]
        graph.output(out)


def test_slice_with_tensor_value(graph_builder) -> None:
    with graph_builder(
        input_types=[
            TensorType(DType.int32, [5, "in_dim"], device=DeviceRef.CPU())
        ],
    ) as graph:
        start = ops.constant(2, DType.int64, device=DeviceRef.CPU())
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
def test_slice_valid_ints(
    graph_builder, tensor_type: TensorType, index
) -> None:
    assume(tensor_type.shape)
    assume(0 not in tensor_type.shape)

    with graph_builder(input_types=[tensor_type]) as graph:
        out = ops.slice_tensor(graph.inputs[0].tensor, index)
        assert out.shape == expected_slice_shape(tensor_type.shape, index)
        graph.output(out)


@given(
    tensor_type=tensor_types(shapes=shared_shapes),
    index=shared_shapes.flatmap(shape_indexes),
)
def test_slice_valid_tensorvalues(
    graph_builder, tensor_type: TensorType, index
) -> None:
    assume(tensor_type.shape)
    assume(0 not in tensor_type.shape)

    with graph_builder(input_types=[tensor_type]) as graph:
        out = ops.slice_tensor(
            graph.inputs[0].tensor,
            [
                ops.constant(i, DType.int64, device=DeviceRef.CPU())
                if isinstance(i, int)
                else i
                for i in list(index)
            ],
        )
        assert out.shape == expected_slice_shape(tensor_type.shape, index)
        graph.output(out)


def gen_slice(n: int, rand: random.Random) -> slice:
    # Assign random start and step with 50% probability else None.
    start = rand.randint(-n, n - 1) if rand.randint(0, 1) else None
    # TODO(AIPIPE-109): allow negative step.
    step = rand.randint(1, n) if rand.randint(0, 1) else None

    # Set default values to compute valid ranges if start/step is None.
    start_val = 0 if start is None else start + n if start < 0 else start
    step_val = 1 if step is None else step

    stop: int | None = None
    if rand.randint(0, 1):
        # Generate a valid, non-empty slice depending on positive/negative step.
        normalized_start = start_val + n if start_val < 0 else start_val
        normalized_stop = (
            rand.randint(normalized_start + 1, n)
            if step_val > 0
            else rand.randint(0, normalized_start - 1)
        )

        # Filter [n:n+1:1] and [-n:-n-1:-1] (index out of bounds).
        assume(normalized_stop < n and normalized_stop > -n - 1)

        stop = normalized_stop if rand.randint(0, 1) else -n + normalized_stop

    # Check for overflow with normalized start/stop.
    stop_val = n if stop is None else stop + n if stop < 0 else stop
    step_offset = step_val - 1 if step_val > 0 else step_val + 1
    diff = stop_val - start_val
    int64_max = np.iinfo(np.int64).max
    int64_min = np.iinfo(np.int64).min

    # Don't generate slices that would overflow --- that is checked elsewhere.
    assume(
        int64_min <= diff <= int64_max
        and int64_min <= diff + step_offset <= int64_max
    )

    return slice(start, stop, step)


static_tensor_type = tensor_types(
    shapes=st.shared(st.from_type(list[StaticDim]))
)


@given(tensor_type=static_tensor_type, rand=...)
def test_slice_static_dims(
    graph_builder, tensor_type: TensorType, rand: random.Random
) -> None:
    assume(tensor_type.shape)
    assume(0 not in tensor_type.shape)

    index = [gen_slice(int(d), rand) for d in tensor_type.shape]

    with graph_builder(input_types=[tensor_type]) as graph:
        out = ops.slice_tensor(graph.inputs[0].tensor, index)
        assert out.shape == expected_slice_shape(tensor_type.shape, index)
        graph.output(out)


@pytest.mark.parametrize(
    ("tensor_type", "indices"),
    [
        # x[1:]
        (
            TensorType(DType.float32, shape=["dim0"], device=DeviceRef.CPU()),
            (slice(1, None),),
        ),
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (slice(1, None),),
        ),
        # x[:-1]
        (
            TensorType(DType.float32, shape=["dim0"], device=DeviceRef.CPU()),
            (slice(None, -1),),
        ),
        # x[::2]
        (
            TensorType(DType.float32, shape=["dim0"], device=DeviceRef.CPU()),
            (slice(None, None, 2),),
        ),
        # x[::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (TensorType(DType.float32, shape=["dim0"]), (slice(None, None, -1),)),
        # x[:, None, :]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (slice(None), None, slice(None)),
        ),
        # x[None, ...]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (None, Ellipsis),
        ),
        # x[..., None]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (Ellipsis, None),
        ),
        # x[..., 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (Ellipsis, 1),
        ),
        # x[Ellipsis, 1:]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (Ellipsis, slice(1, None)),
        ),
        # x[1, ..., ::-1]
        # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
        # (
        #     TensorType(DType.float32, shape=["dim0", "dim1", "dim2"]),
        #     (1, Ellipsis, slice(None, None, -1)),
        # ),
    ],
)
def test_slice_symbolic_tensor(
    graph_builder, tensor_type: TensorType, indices: list[slice]
) -> None:
    """Tests slicing vectors of symbolic dims by another symbolic dim vector."""
    # NOTE: the `Graph` constructor verifies the staged graph op.
    Graph(
        "slice",
        forward=operator.itemgetter(indices),
        input_types=[tensor_type],
    )


@pytest.mark.parametrize(
    ("tensor_type", "indices"),
    [
        (
            TensorType(DType.int32, shape=[1], device=DeviceRef.CPU()),
            (slice(-6618538577426847335, None, 3019951631318595876)),
        ),
    ],
)
def test_slice_dim_overflow(
    graph_builder, tensor_type: TensorType, indices: list[slice]
) -> None:
    """Tests cases that would overflow an int64 slice index."""
    with pytest.raises(
        ValueError, match="rmo.slice index computation overflow"
    ):
        Graph(
            "slice",
            forward=operator.itemgetter(indices),
            input_types=[tensor_type],
        )


@pytest.mark.parametrize(
    ("tensor_type", "indices", "expected_length", "expected_none_indices"),
    [
        # x[:, None, :]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (slice(None), None, slice(None)),
            3,
            (1,),
        ),
        # x[None, ..., None]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (None, Ellipsis, None),
            4,
            (0, 3),
        ),
        # x[..., None]
        (
            TensorType(
                DType.float32, shape=["dim0", "dim1"], device=DeviceRef.CPU()
            ),
            (Ellipsis, None),
            3,
            (2,),
        ),
    ],
)
def test_slice_none_dims(
    graph_builder,
    tensor_type: TensorType,
    indices: list[slice],
    expected_length: int,
    expected_none_indices: tuple[int, ...],
) -> None:
    """Tests slicing vectors of symbolic dims by another symbolic dim vector."""
    # NOTE: the `Graph` constructor verifies the staged graph op.
    graph = Graph(
        "slice",
        forward=operator.itemgetter(indices),
        input_types=[tensor_type],
    )

    (result_type,) = graph.output_types
    # Check that the output rank is correctly expanded by the None indices.
    assert result_type.rank == expected_length

    # Check that all the expanded dims are 1.
    assert all(result_type.shape[i] == 1 for i in expected_none_indices)


@pytest.mark.parametrize(
    ("tensor_type", "indices", "expected_shape"),
    [
        # x[1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (1,),
            ["dim1", "dim2"],
        ),
        # x[:, 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (slice(None), 1),
            ["dim0", "dim2"],
        ),
        # x[:, :, 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (slice(None), slice(None), 1),
            ["dim0", "dim1"],
        ),
        # x[1, 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (1, 1),
            ["dim2"],
        ),
        # x[1, :, 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (1, slice(None), 1),
            ["dim1"],
        ),
        # x[1, 1, 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (1, 1, 1),
            [],
        ),
        # x[..., 1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (Ellipsis, 1),
            ["dim0", "dim1"],
        ),
        # x[1, ...]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (1, Ellipsis),
            ["dim1", "dim2"],
        ),
        # x[:, -1]
        (
            TensorType(
                DType.float32,
                shape=["dim0", "dim1", "dim2"],
                device=DeviceRef.CPU(),
            ),
            (slice(None), -1),
            ["dim0", "dim2"],
        ),
    ],
)
def test_slice_int_dims(
    graph_builder,
    tensor_type: TensorType,
    indices: tuple[Any, ...],
    expected_shape: list[str | int],
) -> None:
    """Tests slicing vectors of symbolic dims by another symbolic dim vector."""
    # NOTE: the `Graph` constructor verifies the staged graph op.
    graph = Graph(
        "slice",
        forward=operator.itemgetter(indices),
        input_types=[tensor_type],
    )
    (result_type,) = graph.output_types

    # Check that the output rank is correctly expanded by the None indices.
    assert result_type.rank == len(expected_shape)
    assert all(
        dim == expected_dim
        for dim, expected_dim in zip(result_type.shape, expected_shape)
        if isinstance(expected_dim, int)
    )


def test_slice_invalid_start_stop(graph_builder) -> None:
    """Checks that slicing with invalid start/stop/step raises an error."""
    input_type = TensorType(
        DType.float32, shape=["dim0"], device=DeviceRef.CPU()
    )
    with graph_builder(input_types=[input_type]) as graph:
        x = graph.inputs[0]
        with pytest.raises(
            ValueError,
            match=(
                "start and stop should be increasing for positive step and "
                "decreasing for negative step, but got start 2, stop 1 for step 1"
            ),
        ):
            x[2:1]
