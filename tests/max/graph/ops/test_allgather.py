# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from conftest import tensor_types
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType, ops

shared_types = st.shared(tensor_types())


def test_allgather_rep_device() -> None:
    """Test unique device error for allgather."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "allgather operation must have unique devices across its input"
            " tensors."
        ),
    ):
        with Graph(
            "allgather",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
            ],
        ) as graph:
            allgather_outputs = ops.allgather(v.tensor for v in graph.inputs)
            graph.output(
                allgather_outputs[0],
                allgather_outputs[1],
                allgather_outputs[2],
                allgather_outputs[3],
            )


def test_allgather_wrong_shape() -> None:
    """Test wrong shape error for allgather when non-concat dimensions don't match."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "allgather operation inputs must have the same shape in all"
            " dimensions except the concatenation dimension"
        ),
    ):
        with Graph(
            "allgather",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 2], device=devices[1]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[2]
                ),
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[3]
                ),
            ],
        ) as graph:
            allgather_outputs = ops.allgather(v.tensor for v in graph.inputs)

            graph.output(
                allgather_outputs[0],
                allgather_outputs[1],
                allgather_outputs[2],
                allgather_outputs[3],
            )


def test_allgather_uneven_shapes() -> None:
    """Test allgather with uneven shapes along concatenation dimension."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    # Test with uneven shapes along dimension 0.
    with Graph(
        "allgather_uneven",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[37919, 4096], device=devices[0]
            ),
            TensorType(
                dtype=DType.float32, shape=[37919, 4096], device=devices[1]
            ),
            TensorType(
                dtype=DType.float32, shape=[37918, 4096], device=devices[2]
            ),
            TensorType(
                dtype=DType.float32, shape=[37918, 4096], device=devices[3]
            ),
        ],
    ) as graph:
        allgather_outputs = ops.allgather(v.tensor for v in graph.inputs)

        # Check output shapes - should be sum of input shapes along axis 0.
        expected_dim0 = 37919 + 37919 + 37918 + 37918  # 151674
        for output in allgather_outputs:
            assert len(output.shape) == 2
            assert int(output.shape[0]) == expected_dim0
            assert int(output.shape[1]) == 4096

        graph.output(*allgather_outputs)


def test_allgather_bad_dim() -> None:
    """Test wrong shape error for allgather."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[3]),
        ],
    ) as graph:
        with pytest.raises(IndexError, match="Dimension out of range"):
            _ = ops.allgather((v.tensor for v in graph.inputs), axis=-3)

        with pytest.raises(IndexError, match="Dimension out of range"):
            _ = ops.allgather((v.tensor for v in graph.inputs), axis=-5)


def test_allgather_basic() -> None:
    """Test basic allgather use case."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[3]),
        ],
    ) as graph:
        allgather_outputs = ops.allgather(v.tensor for v in graph.inputs)
        graph.output(
            allgather_outputs[0],
            allgather_outputs[1],
            allgather_outputs[2],
            allgather_outputs[3],
        )
        for output, device in zip(allgather_outputs, devices):
            assert device == output.device
            assert output.shape == Shape((24, 5))
            assert output.dtype == DType.float32


def test_allgather_nonzero_dim() -> None:
    """Test allgather with non-default axis concatenation."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[3]),
        ],
    ) as graph:
        outputs = ops.allgather((v.tensor for v in graph.inputs), axis=-2)
        for output in outputs:
            assert output.shape == Shape((6, 20, 4))

        outputs_dim_1 = ops.allgather((v.tensor for v in graph.inputs), axis=1)
        for output in outputs_dim_1:
            assert output.shape == Shape((6, 20, 4))

        outputs_dim_2 = ops.allgather((v.tensor for v in graph.inputs), axis=2)
        for output in outputs_dim_2:
            assert output.shape == Shape((6, 5, 16))


def test_allgather_nonzero_dim_uneven() -> None:
    """Test allgather with non-default axis concatenation and uneven shapes."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]

    # Test with uneven shapes along dimension 1.
    with Graph(
        "allgather_uneven_dim1",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[6, 5, 4], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[6, 4, 4], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 4, 4], device=devices[3]),
        ],
    ) as graph:
        outputs = ops.allgather((v.tensor for v in graph.inputs), axis=1)
        for output in outputs:
            assert output.shape == Shape((6, 18, 4))  # 5+5+4+4=18

        graph.output(*outputs)

    # Test with uneven shapes along dimension 2.
    with Graph(
        "allgather_uneven_dim2",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5, 10], device=devices[0]
            ),
            TensorType(
                dtype=DType.float32, shape=[6, 5, 10], device=devices[1]
            ),
            TensorType(dtype=DType.float32, shape=[6, 5, 9], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[6, 5, 9], device=devices[3]),
        ],
    ) as graph:
        outputs = ops.allgather((v.tensor for v in graph.inputs), axis=2)
        for output in outputs:
            assert output.shape == Shape((6, 5, 38))  # 10+10+9+9=38

        graph.output(*outputs)


def test_allgather_noop() -> None:
    """Tests that allgather is a noop if the number of inputs is 0 or 1."""
    with Graph(
        "allgather",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.GPU(id=0)
            ),
        ],
    ) as graph:
        allgather_outputs = ops.allgather([graph.inputs[0].tensor])
        assert allgather_outputs[0] is graph.inputs[0].tensor

        allgather_outputs = ops.allgather([])
        assert not allgather_outputs
