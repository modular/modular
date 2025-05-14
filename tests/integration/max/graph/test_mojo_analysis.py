# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings for mojo analysis."""

import os
from pathlib import Path

import pytest
from max import mlir
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, _OpaqueType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_kernel_library(counter_mojopkg, kernel_verification_ops_path):
    with Graph("test_kernel_library") as graph:
        kernels = graph._kernel_library
        kernels.add_path(counter_mojopkg)

        assert "make_counter" in kernels
        assert isinstance(kernels["make_counter"], mlir.Operation)
        assert "my_add" not in kernels

        with pytest.raises(ValueError) as err:
            kernels.add_path(Path("/path/to/invalid.mojopkg"))
        assert "No such file or directory" in str(err.value)

        kernels.add_path(kernel_verification_ops_path)
        assert "make_counter" in kernels
        assert "my_add" in kernels
        assert isinstance(kernels["my_add"], mlir.Operation)


def test_undefined_kernel(kernel_verification_ops_path: Path) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_undefined_kernel",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        with pytest.raises(ValueError) as err:
            ops.custom(
                "i_am_not_a_kernel",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )
        assert (
            "Could not find a mojo kernel registered for i_am_not_a_kernel"
            in str(err.value)
        )


def test_my_add_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_my_add_valid",
        input_types=[tensor_type, tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            ops.custom(
                "my_add",
                values=[graph.inputs[0], graph.inputs[1]],
                out_types=[tensor_type],
            )[0]
        )

    # Compile the model
    session.load(graph)


def test_my_add_invalid_inputs_count(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_my_add_invalid_inputs_count",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        with pytest.raises(ValueError) as err:
            ops.custom(
                "my_add",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )
        assert (
            "Could not provide all input arguments to custom op 'my_add': missing operands starting from argument named 'y'"
            in str(err.value)
        )


def test_my_add_invalid_outputs_count(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_my_add_invalid_outputs_count",
        input_types=[tensor_type, tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        with pytest.raises(ValueError) as err:
            ops.custom(
                "my_add",
                values=[graph.inputs[0], graph.inputs[1]],
                out_types=[tensor_type, tensor_type],
            )
        assert (
            "Custom op 'my_add' has too many results: results starting at position 1 have no equivalent output argument or function result in kernel"
            in str(err.value)
        )


def test_op_with_device_context_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_with_device_context_valid",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            ops.custom(
                "op_with_device_context",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )[0]
        )

    # Compile the model
    session.load(graph)


def test_op_multiple_outputs_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_multiple_outputs_valid",
        input_types=[tensor_type],
        output_types=[tensor_type, tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            *ops.custom(
                "op_with_multiple_outputs",
                values=[graph.inputs[0]],
                out_types=[tensor_type, tensor_type],
            )
        )

    # Compile the model
    session.load(graph)


def test_op_multiple_outputs_invalid_outputs_count(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_multiple_outputs_invalid_outputs_count",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        with pytest.raises(ValueError) as err:
            ops.custom(
                "op_with_multiple_outputs",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )
        assert (
            "Could not provide all output arguments to custom op 'op_with_multiple_outputs': missing operands starting from argument named 'out1'"
            in str(err.value)
        )


def test_op_without_outputs_invalid_outputs_count(
    kernel_verification_ops_path: Path,
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_op_without_outputs_invalid_outputs_count",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])

        with pytest.raises(ValueError) as err:
            ops.custom(
                "op_without_outputs",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )
        assert (
            "Custom op 'op_without_outputs' has too many results: results starting at position 0 have no equivalent output argument or function result in kernel"
            in str(err.value)
        )


def test_return_opaque_mem_type(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    opaque_type = _OpaqueType("MyIntMemory")
    graph = Graph(
        "test_return_opaque_mem_type",
        input_types=[tensor_type],
        output_types=[opaque_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            ops.custom(
                "make_my_int_memory",
                values=[graph.inputs[0]],
                out_types=[opaque_type],
            )[0]
        )

    # Compile the model
    session.load(graph)


def test_return_opaque_reg_type(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.int32, [1], device=DeviceRef.CPU())
    opaque_type = _OpaqueType("MyIntReg")
    graph = Graph(
        "test_return_opaque_reg_type",
        input_types=[tensor_type],
        output_types=[opaque_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            ops.custom(
                "make_my_int_reg",
                values=[graph.inputs[0]],
                out_types=[opaque_type],
            )[0]
        )

    # Compile the model
    session.load(graph)


def test_variadic_ins_outs_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_variadic_ins_outs_valid",
        input_types=[tensor_type, tensor_type, tensor_type],
        output_types=[tensor_type, tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            *ops.custom(
                "variadic_input_to_output",
                values=[graph.inputs[0], graph.inputs[1], graph.inputs[2]],
                out_types=[tensor_type, tensor_type],
            )
        )

    # Compile the model
    session.load(graph)


def test_variadic_size_0_invalid(kernel_verification_ops_path: Path) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_variadic_size_0_invalid",
        input_types=[tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        with pytest.raises(ValueError) as err:
            ops.custom(
                "variadic_add",
                values=[graph.inputs[0]],
                out_types=[tensor_type],
            )
        assert (
            "Could not provide all input arguments to custom op 'variadic_add': missing operands starting from argument named 'input'"
            in str(err.value)
        )


def test_tensor_kernel_raises_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    tensor_type = TensorType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph(
        "test_tensor_kernel_raises_valid",
        input_types=[tensor_type, tensor_type],
        output_types=[tensor_type],
    )
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        graph.output(
            ops.custom(
                "binary_kernel_with_raises",
                values=[graph.inputs[0], graph.inputs[1]],
                out_types=[tensor_type],
            )[0]
        )

    # Compile the model
    session.load(graph)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="TODO(GEX-2136): Graph generating erroneous transfer to cpu for buffer",
)
def test_mutable_input_tensor_valid(
    session: InferenceSession, kernel_verification_ops_path: Path
) -> None:
    buffer_type = BufferType(DType.float32, [64], device=DeviceRef.CPU())
    graph = Graph("test_mutable_input_tensor_valid", input_types=[buffer_type])
    with graph:
        graph._import_kernels([kernel_verification_ops_path])
        ops.inplace_custom(
            "mutable_input_tensor",
            values=[graph.inputs[0]],
        )
        graph.output()

    # Compile the model
    session.load(graph)
