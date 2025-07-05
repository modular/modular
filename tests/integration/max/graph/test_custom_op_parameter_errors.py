# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
            match="""failed to convert custom op 'op_with_int_parameter' attribute 'IntParameter' with value '"Not an int!"' to the proper Mojo parameter type: expected an integer type""",
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
            match="failed to convert custom op 'op_with_int_parameter' attribute 'IntParameter' with value 'si32' to the proper Mojo parameter type: expected an integer type",
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
            match="failed to convert custom op 'op_with_dtype_parameter' attribute 'DTypeParameter' with value '\"Not a dtype!\"' to the proper Mojo parameter type: expected a DType",
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
            match="failed to convert custom op 'op_with_dtype_parameter' attribute 'DTypeParameter' with value '42 : index' to the proper Mojo parameter type: expected a DType",
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
            match="in custom op 'op_with_string_parameter' attribute 'StringParameter' uses type String. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="in custom op 'op_with_string_parameter' attribute 'StringParameter' uses type String. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="in custom op 'op_with_string_parameter' attribute 'StringParameter' uses type String. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="in custom op 'op_with_string_slice_parameter' attribute 'StringParameter' uses type StringSlice. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="in custom op 'op_with_string_slice_parameter' attribute 'StringParameter' uses type StringSlice. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="in custom op 'op_with_string_slice_parameter' attribute 'StringParameter' uses type StringSlice. Use type StaticString instead to define a string parameter for a custom op.",
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
            match="failed to convert custom op 'op_with_static_string_parameter' attribute 'StringParameter' with value '42 : index' to the proper Mojo parameter type: expected string type",
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
        with pytest.raises(
            ValueError,
            match="parameter 'UnusedParameter' supplied to custom op 'op_with_int_parameter' does not exist in the corresponding Mojo code",
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
        with pytest.raises(
            ValueError,
            match="parameter 'UnusedParameter' supplied to custom op 'op_with_dtype_parameter' does not exist in the corresponding Mojo code",
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
        with pytest.raises(
            ValueError,
            match="parameter 'UnusedParameter' supplied to custom op 'op_with_static_string_parameter' does not exist in the corresponding Mojo code",
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
        with pytest.raises(
            ValueError,
            match="parameter 'UnusedParam1' supplied to custom op 'op_with_int_parameter' does not exist in the corresponding Mojo code",
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
            match="failed to convert custom op 'op_with_static_string_parameter' attribute 'StringParameter' with value 'si32' to the proper Mojo parameter type: expected string type",
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
