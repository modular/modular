# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.range tests."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType, ops


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (0, 5, 1),  # normal positive step, start < stop
        (5, 0, -1),  # normal negative step, start > stop
        (0, 0, 1),  # start == stop (empty range)
        (-5, 0, 1),  # negative start, positive step
        (0, -5, -1),  # negative step, start > stop
        (0, 5, 2),  # step does not evenly divide into stop - start
        (0, 5, 10),  # step larger than stop - start
        (0, 2**63 - 1, 1),  # full int64 range
        (-1, -(2**63), -1),  # full int64 range, negative step
    ],
)
def test_range(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should pass."""
    with Graph("range", input_types=()) as graph:
        dim = (stop - start) // step
        start_val = ops.constant(start, DType.int64, device=DeviceRef.CPU())
        stop_val = ops.constant(stop, DType.int64, device=DeviceRef.CPU())
        step_val = ops.constant(step, DType.int64, device=DeviceRef.CPU())
        out = ops.range(
            start_val, stop_val, step_val, out_dim=dim, device=DeviceRef.CPU()
        )
        graph.output(out)
        assert out.shape == Shape(
            [
                dim,
            ]
        )


# Strategy for valid range dtypes (numeric types)
range_dtypes = st.sampled_from(
    [
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.float16,
        DType.float32,
        DType.float64,
    ]
)

# Strategy for devices
devices = st.sampled_from([DeviceRef.CPU(), DeviceRef.GPU()])


# Strategy for valid range parameters that produce reasonable output sizes
@st.composite
def valid_range_params(draw):
    """Generate valid range parameters with reasonable output dimensions."""
    # Generate step first (non-zero)
    step = draw(
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    )

    # Generate start and stop that produce reasonable dim (1-1000)
    if step > 0:
        start = draw(st.integers(min_value=0, max_value=500))
        dim = draw(st.integers(min_value=1, max_value=1000))
        stop = start + (dim * step)
    else:
        start = draw(st.integers(min_value=500, max_value=1000))
        dim = draw(st.integers(min_value=1, max_value=1000))
        stop = start + (dim * step)

    return start, stop, step, dim


@given(dtype=range_dtypes)
def test_range_dtypes_hypothesis(dtype: DType) -> None:
    """Tests ops.range with different data types using hypothesis."""
    with Graph("range_dtypes", input_types=()) as graph:
        start_val = ops.constant(0, dtype, device=DeviceRef.CPU())
        stop_val = ops.constant(5, dtype, device=DeviceRef.CPU())
        step_val = ops.constant(1, dtype, device=DeviceRef.CPU())
        out = ops.range(
            start_val, stop_val, step_val, out_dim=5, device=DeviceRef.CPU()
        )
        graph.output(out)
        assert out.shape == Shape([5])
        assert out.dtype == dtype


@given(device=devices)
def test_range_devices_hypothesis(device: DeviceRef) -> None:
    """Tests ops.range with different device types using hypothesis."""
    with Graph("range_devices", input_types=()) as graph:
        start_val = ops.constant(0, DType.int32, device=DeviceRef.CPU())
        stop_val = ops.constant(3, DType.int32, device=DeviceRef.CPU())
        step_val = ops.constant(1, DType.int32, device=DeviceRef.CPU())
        out = ops.range(start_val, stop_val, step_val, out_dim=3, device=device)
        graph.output(out)
        assert out.shape == Shape([3])
        assert out.device == device


def test_range_tensor_inputs() -> None:
    """Tests ops.range with tensor value inputs (not constants)."""
    with Graph(
        "range_tensor_inputs",
        input_types=(
            TensorType(shape=(), dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=(), dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=(), dtype=DType.int32, device=DeviceRef.CPU()),
        ),
    ) as graph:
        start_val = graph.inputs[0]
        stop_val = graph.inputs[1]
        step_val = graph.inputs[2]
        out = ops.range(
            start_val, stop_val, step_val, out_dim=10, device=DeviceRef.CPU()
        )
        graph.output(out)
        assert out.shape == Shape([10])


@pytest.mark.parametrize(
    "start, stop, step",
    [
        # TODO(bduke): step == 0 should raise in the range op builder.
        # (0, 5, 0),  # step = 0
        (0, 2**63, 1),  # dim exceeds int64 max
        (0, -(2**63), -1),  # dim exceeds int64 max in negative direction
        (2**62, 2**63, 1),  # large range, dim exceeds int64 max
        (-(2**63), 0, 1),  # large negative start
    ],
)
def test_range_exceptions(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should raise an exception."""
    with pytest.raises(ValueError):
        with Graph("range", input_types=()) as graph:
            # Set dim to 0 as a placeholder when we would divide by zero.
            dim = (stop - start) // step if step != 0 else 0
            start_val = ops.constant(start, DType.int64, device=DeviceRef.CPU())
            stop_val = ops.constant(stop, DType.int64, device=DeviceRef.CPU())
            step_val = ops.constant(step, DType.int64, device=DeviceRef.CPU())
            out = ops.range(
                start_val,
                stop_val,
                step_val,
                dim,
                device=DeviceRef.CPU(),
                dtype=DType.int64,
            )
            graph.output(out)


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (0, 5, 1),  # normal positive step, start < stop
        (5, 0, -1),  # normal negative step, start > stop
        (0, 0, 1),  # start == stop (empty range)
        (-5, 0, 1),  # negative start, positive step
        (0, -5, -1),  # negative step, start > stop
        (0, 5, 2),  # step does not evenly divide into stop - start
        (0, 5, 10),  # step larger than stop - start
        (0, 2**63 - 1, 1),  # full int64 range
        (-1, -(2**63), -1),  # full int64 range, negative step
    ],
)
def test_range_numeric(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should pass."""
    with Graph("range", input_types=()) as graph:
        dim = (stop - start) // step if step != 0 else 0
        out = ops.range(start, stop, step, device=DeviceRef.CPU())
        graph.output(out)
        assert out.shape == Shape(
            [
                dim,
            ]
        )


@pytest.mark.parametrize(
    "start, stop, step",
    [
        # TODO(bduke): step == 0 should raise in the range op builder.
        # (0, 5, 0),  # step = 0
        (0, 2**63, 1),  # dim exceeds int64 max
        (0, -(2**63), -1),  # dim exceeds int64 max in negative direction
        (2**62, 2**63, 1),  # large range, dim exceeds int64 max
        (-(2**63), 0, 1),  # large negative start
    ],
)
def test_range_exceptions_numeric(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should raise an exception."""
    with pytest.raises(ValueError):
        with Graph("range", input_types=()) as graph:
            out = ops.range(
                start,
                stop,
                step,
                device=DeviceRef.CPU(),
                dtype=DType.int64,
            )
            graph.output(out)


@given(dtype1=range_dtypes, dtype2=range_dtypes)
def test_range_dtype_mismatch(dtype1: DType, dtype2: DType) -> None:
    """Tests ops.range with mismatched dtypes raises ValueError using hypothesis."""
    assume(dtype1 != dtype2)  # Only test when dtypes are actually different
    with pytest.raises(
        ValueError, match="range expected inputs of the same type!"
    ):
        with Graph("range_dtype_mismatch", input_types=()) as graph:
            start_val = ops.constant(0, dtype1, device=DeviceRef.CPU())
            stop_val = ops.constant(
                5, dtype2, device=DeviceRef.CPU()
            )  # Different dtype
            step_val = ops.constant(1, dtype1, device=DeviceRef.CPU())
            out = ops.range(
                start_val, stop_val, step_val, out_dim=5, device=DeviceRef.CPU()
            )
            graph.output(out)


def test_range_dtype_mismatch_specific_error() -> None:
    """Tests ops.range with mismatched dtypes raises ValueError."""
    with pytest.raises(
        ValueError, match="range expected inputs of the same type!"
    ):
        with Graph("range_dtype_mismatch", input_types=()) as graph:
            start_val = ops.constant(0, DType.int32, device=DeviceRef.CPU())
            stop_val = ops.constant(
                5, DType.float32, device=DeviceRef.CPU()
            )  # Different dtype
            step_val = ops.constant(1, DType.int32, device=DeviceRef.CPU())
            out = ops.range(
                start_val, stop_val, step_val, out_dim=5, device=DeviceRef.CPU()
            )
            graph.output(out)


def test_range_non_scalar_inputs_specific_error() -> None:
    """Tests ops.range with non-scalar inputs raises ValueError."""
    with pytest.raises(
        ValueError, match="range expected scalar values as inputs!"
    ):
        with Graph("range_non_scalar", input_types=()) as graph:
            start_val = ops.constant(
                np.array([0]), DType.int32, device=DeviceRef.CPU()
            )  # Non-scalar
            stop_val = ops.constant(5, DType.int32, device=DeviceRef.CPU())
            step_val = ops.constant(1, DType.int32, device=DeviceRef.CPU())
            out = ops.range(
                start_val, stop_val, step_val, out_dim=5, device=DeviceRef.CPU()
            )
            graph.output(out)


def test_range_missing_out_dim_with_dynamic_inputs() -> None:
    """Tests ops.range with dynamic inputs and no out_dim raises ValueError."""
    with pytest.raises(
        ValueError,
        match="range expected out_dim for non-numeric values as inputs!",
    ):
        with Graph(
            "range_missing_out_dim",
            input_types=(
                TensorType(shape=(), dtype=DType.int32, device=DeviceRef.CPU()),
            ),
        ) as graph:
            start_val = graph.inputs[0]  # Dynamic input
            stop_val = ops.constant(5, DType.int32, device=DeviceRef.CPU())
            step_val = ops.constant(1, DType.int32, device=DeviceRef.CPU())
            out = ops.range(
                start_val,
                stop_val,
                step_val,
                device=DeviceRef.CPU(),  # No out_dim provided
            )
            graph.output(out)


def test_range_step_zero() -> None:
    """Tests ops.range with step == 0."""
    # Note: This currently doesn't raise at op construction time but should
    # raise at execution time. Testing the current behavior.
    with Graph("range_step_zero", input_types=()) as graph:
        start_val = ops.constant(0, DType.int32, device=DeviceRef.CPU())
        stop_val = ops.constant(5, DType.int32, device=DeviceRef.CPU())
        step_val = ops.constant(
            0, DType.int32, device=DeviceRef.CPU()
        )  # step = 0
        out = ops.range(
            start_val, stop_val, step_val, out_dim=0, device=DeviceRef.CPU()
        )
        graph.output(out)
        # Currently this passes graph construction but would fail at execution
        assert out.shape == Shape([0])


# Strategy for sign mismatch cases
@st.composite
def sign_mismatch_params(draw):
    """Generate parameters where sign(stop-start) != sign(step)."""
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        # positive step but start > stop
        start = draw(st.integers(min_value=1, max_value=100))
        stop = draw(st.integers(min_value=-100, max_value=start - 1))
        step = draw(st.integers(min_value=1, max_value=10))
    elif choice == 1:
        # negative step but start < stop
        start = draw(st.integers(min_value=-100, max_value=99))
        stop = draw(st.integers(min_value=start + 1, max_value=100))
        step = draw(st.integers(min_value=-10, max_value=-1))
    else:
        # another negative step but start < stop case
        start = draw(st.integers(min_value=-100, max_value=-1))
        stop = draw(st.integers(min_value=0, max_value=100))
        step = draw(st.integers(min_value=-10, max_value=-1))

    return start, stop, step


@given(params=sign_mismatch_params())
def test_range_sign_mismatch(params) -> None:
    """Tests ops.range where sign(stop-start) != sign(step) using hypothesis."""
    start, stop, step = params
    # Note: This currently doesn't raise at op construction time but should
    # produce empty ranges or raise at execution time.
    with Graph("range_sign_mismatch", input_types=()) as graph:
        dim = max(0, (stop - start) // step) if step != 0 else 0
        start_val = ops.constant(start, DType.int32, device=DeviceRef.CPU())
        stop_val = ops.constant(stop, DType.int32, device=DeviceRef.CPU())
        step_val = ops.constant(step, DType.int32, device=DeviceRef.CPU())
        out = ops.range(
            start_val, stop_val, step_val, out_dim=dim, device=DeviceRef.CPU()
        )
        graph.output(out)
        # Currently this passes graph construction but produces empty range
        assert out.shape == Shape([dim])


@given(dtype=range_dtypes, device=devices)
def test_range_valid_params(dtype: DType, device: DeviceRef) -> None:
    """Tests ops.range with valid parameters using hypothesis fuzzing."""
    # Use safe small values that work with all dtypes
    start, stop, step, expected_dim = 0, 10, 1, 10
    with Graph("range_valid", input_types=()) as graph:
        start_val = ops.constant(start, dtype, device=DeviceRef.CPU())
        stop_val = ops.constant(stop, dtype, device=DeviceRef.CPU())
        step_val = ops.constant(step, dtype, device=DeviceRef.CPU())
        out = ops.range(
            start_val, stop_val, step_val, out_dim=expected_dim, device=device
        )
        graph.output(out)
        assert out.shape == Shape([expected_dim])
        assert out.dtype == dtype
        assert out.device == device


# Strategy for testing non-scalar inputs
@st.composite
def non_scalar_arrays(draw):
    """Generate non-scalar numpy arrays for testing."""
    shape = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3)
    )
    size = 1
    for dim in shape:
        size *= dim
    # Generate values that fit in the shape
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=10), min_size=size, max_size=size
        )
    )
    return np.array(values).reshape(shape)


@given(array=non_scalar_arrays())
def test_range_non_scalar_inputs(array) -> None:
    """Tests ops.range with non-scalar inputs raises ValueError using hypothesis."""
    with pytest.raises(
        ValueError, match="range expected scalar values as inputs!"
    ):
        with Graph("range_non_scalar", input_types=()) as graph:
            start_val = ops.constant(
                array, DType.int32, device=DeviceRef.CPU()
            )  # Non-scalar
            stop_val = ops.constant(5, DType.int32, device=DeviceRef.CPU())
            step_val = ops.constant(1, DType.int32, device=DeviceRef.CPU())
            out = ops.range(
                start_val, stop_val, step_val, out_dim=5, device=DeviceRef.CPU()
            )
            graph.output(out)
