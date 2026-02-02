# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Test the error messages for custom ops related to parameters."""

import os
from pathlib import Path

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_op_with_int_parameter_passed_as_string(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_int_parameter_passed_as_string",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_int_parameter' attribute 'IntParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_int_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"IntParameter": "Not an int!"},
                )[0]
            )


def test_op_with_int_parameter_passed_as_dtype(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_int_parameter_passed_as_dtype",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_int_parameter' attribute 'IntParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_int_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"IntParameter": DType.int32},
                )[0]
            )


def test_op_with_dtype_parameter_passed_as_string(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_dtype_parameter_passed_as_string",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_dtype_parameter' attribute 'DTypeParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_dtype_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"DTypeParameter": "Not a dtype!"},
                )[0]
            )


def test_op_with_dtype_parameter_passed_as_int(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_dtype_parameter_passed_as_int",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_dtype_parameter' attribute 'DTypeParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_dtype_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"DTypeParameter": 42},
                )[0]
            )


def test_op_with_string_parameter_passed_as_int(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_parameter_passed_as_int",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": 42},
                )[0]
            )


def test_op_with_string_parameter_passed_as_dtype(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_parameter_passed_as_dtype",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": DType.int32},
                )[0]
            )


def test_op_with_string_parameter_passed_as_string_literal(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_parameter_passed_as_string_literal",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": "String literal"},
                )[0]
            )


def test_op_with_string_slice_parameter_passed_as_int(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_slice_parameter_passed_as_int",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_slice_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_slice_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": 42},
                )[0]
            )


def test_op_with_string_slice_parameter_passed_as_dtype(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_slice_parameter_passed_as_dtype",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_slice_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_slice_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": DType.int32},
                )[0]
            )


def test_op_with_string_slice_parameter_passed_as_string_literal(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_string_slice_parameter_passed_as_string_literal",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_string_slice_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_string_slice_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": "String literal"},
                )[0]
            )


def test_op_with_static_string_parameter_passed_as_int(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_static_string_parameter_passed_as_int",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_static_string_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_static_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": 42},
                )[0]
            )


def test_op_with_unused_parameter_int(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_unused_parameter_int",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        # Test that validation catches either the unused parameter or a type
        # mismatch (validation order may differ between implementations)
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_int_parameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_int_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"IntParameter": 42, "UnusedParameter": 123},
                )[0]
            )


def test_op_with_unused_parameter_dtype(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_unused_parameter_dtype",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        # Test that validation catches either the unused parameter or a type
        # mismatch (validation order may differ between implementations)
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_dtype_parameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_dtype_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={
                        "DTypeParameter": DType.float32,
                        "UnusedParameter": DType.int64,
                    },
                )[0]
            )


def test_op_with_unused_parameter_string(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_unused_parameter_string",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        # Test that validation catches either the unused parameter or a type
        # mismatch (validation order may differ between implementations)
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_static_string_parameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_static_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={
                        "StringParameter": "valid_string",
                        "UnusedParameter": "unused_value",
                    },
                )[0]
            )


def test_op_with_multiple_unused_parameters(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_multiple_unused_parameters",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        # Test that validation catches either the unused parameters or a type
        # mismatch (validation order may differ between implementations)
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_int_parameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_int_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={
                        "IntParameter": 42,
                        "UnusedParam1": "unused1",
                        "UnusedParam2": 999,
                        "UnusedParam3": DType.float16,
                    },
                )[0]
            )


def test_op_with_static_string_parameter_passed_as_dtype(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_static_string_parameter_passed_as_dtype",
        input_types=[tensor_type],
        output_types=[tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        with pytest.raises(
            ValueError,
            match=r".*custom op 'op_with_static_string_parameter' attribute 'StringParameter'.*",
        ):
            graph.output(
                ops.custom(
                    "op_with_static_string_parameter",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[tensor_type],
                    parameters={"StringParameter": DType.int32},
                )[0]
            )
