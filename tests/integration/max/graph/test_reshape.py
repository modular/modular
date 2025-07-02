# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType


@pytest.mark.skip("MAXPLAT-XXX: parameter with no declaration")
def test_MAXPLAT_328__not_divisible_by_4(session):
    input = Tensor(DType.float32, [7, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 4])
        n_patches, _ = x.shape
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    with pytest.raises(Exception):
        model.execute(input)


@pytest.mark.skip("MAXPLAT-XXX: parameter with no declaration")
def test_MAXPLAT_328__divisible_by_4(session):
    input = Tensor(DType.float32, [8, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        x = x.tensor.rebind([Dim("n_patches_over_4") * 4, 4])
        n_patches, _ = x.shape
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    result = model.execute(input)[0]
    assert result.shape == (2, 4, 4)


def test_MAXPLAT_328__no_new_parameter__not_divisible_by_4(session):
    input = Tensor(DType.float32, [7, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        x = x.tensor.rebind([(n_patches // 4) * 4, 4])
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    with pytest.raises(Exception):
        model.execute(input)


def test_MAXPLAT_328__no_new_parameter__divisible_by_4(session):
    input = Tensor(DType.float32, [8, 4], device=CPU())
    input_type = TensorType(DType.float32, ["batch", 4], device=DeviceRef.CPU())

    with Graph("reshape", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        n_patches, _ = x.tensor.shape
        x = x.tensor.rebind([(n_patches // 4) * 4, 4])
        graph.output(x.reshape([n_patches // 4, 4, 4]))

    model = session.load(graph)
    result = model.execute(input)[0]
    assert result.shape == (2, 4, 4)
