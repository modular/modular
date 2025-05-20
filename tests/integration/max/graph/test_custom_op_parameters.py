# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the behavior of custom op parameters."""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_custom_op_with_int_parameter(
    kernel_verification_ops_path: Path, session, capfd
) -> None:
    expected_int = 42

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Tensor.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_int_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        graph.output(
            ops.custom(
                "op_with_int_parameter",
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"IntParameter": expected_int},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert str(expected_int) in execution_output.out


def test_custom_op_with_dtype_parameter(
    kernel_verification_ops_path: Path, session, capfd
) -> None:
    expected_dtype = DType.int32
    expected_dtype_str = "int32"

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Tensor.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_dtype_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        graph.output(
            ops.custom(
                "op_with_dtype_parameter",
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"DTypeParameter": expected_dtype},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert expected_dtype_str in execution_output.out


def test_custom_op_with_static_string_parameter(
    kernel_verification_ops_path: Path, session, capfd
) -> None:
    expected_string = "Socrates is a man"

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Tensor.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_static_string_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        graph.output(
            ops.custom(
                "op_with_static_string_parameter",
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"StringParameter": expected_string},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert expected_string in execution_output.out
