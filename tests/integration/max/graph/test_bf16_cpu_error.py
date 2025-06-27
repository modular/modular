# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import platform

import pytest
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


@pytest.mark.skipif(
    platform.machine() not in ["arm64", "aarch64"],
    reason="BF16 is only unsupported on ARM CPU architecture",
)
def test_bf16_cpu_input_error(session) -> None:
    input_type = TensorType(
        dtype=DType.bfloat16, shape=["dim"], device=DeviceRef.CPU()
    )
    output_type = DType.float32
    with Graph("cast", input_types=[input_type]) as graph:
        graph.output(graph.inputs[0].tensor.cast(output_type))

    with pytest.raises(
        ValueError,
        match="The bf16 data type is not supported on device 'cpu:0'.",
    ):
        session.load(graph)


@pytest.mark.skipif(
    accelerator_api() != "cuda",
    reason="This test is checking if the PTX output is correct, it will be the "
    "same logic for HIP but we need to generalize the asserts.",
)
@pytest.mark.skipif(
    platform.machine() not in ["arm64", "aarch64"],
    reason="BF16 is only unsupported on ARM CPU architecture",
)
def test_bf16_cpu_output_error(session) -> None:
    input_type = TensorType(
        dtype=DType.float32, shape=["dim"], device=DeviceRef.CPU()
    )
    output_type = DType.bfloat16
    with Graph("cast", input_types=[input_type]) as graph:
        graph.output(graph.inputs[0].tensor.cast(output_type))

    with pytest.raises(
        ValueError,
        match="The bf16 data type is not supported on device 'cpu:0'.",
    ):
        session.load(graph)
