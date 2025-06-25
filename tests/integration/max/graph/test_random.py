# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import platform

import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize(
    "dtype", [DType.float64, DType.float32, DType.bfloat16]
)
def test_random_uniform(session: InferenceSession, dtype: DType) -> None:
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    tensor_type = TensorType(dtype, [10, 10], device=DeviceRef.CPU())

    with Graph(
        "random_uniform",
        input_types=[],
    ) as graph:
        ops.random.set_seed(0)

        lower_bound = ops.constant(0.0, dtype, device=DeviceRef.CPU())
        upper_bound = ops.constant(1.0, dtype, device=DeviceRef.CPU())

        output = ops.random.uniform(tensor_type, (lower_bound, upper_bound))

        # Convert back to f32 since numpy does not support bf16
        if dtype == DType.bfloat16:
            output = ops.cast(output, DType.float32)

        graph.output(output)

    model = session.load(graph)
    result = model.execute()[0]
    assert isinstance(result, Tensor)
    array = result.to_numpy()
    assert (array >= 0.0).all() and (array <= 1.0).all()
