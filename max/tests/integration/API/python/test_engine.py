# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with MOF."""

import os
from dataclasses import dataclass
from math import isclose
from pathlib import Path

import numpy as np
import pytest
import torch
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model, TensorSpec, TorchInputSpec
from max.graph import DeviceRef, Graph, TensorType, Value
from max.mlir.dialects import mo

DYLIB_FILE_EXTENSION = "dylib" if os.uname().sysname == "Darwin" else "so"


# This path is used in skipif clauses rather than tests, so we can neither mark
# it as a fixture nor can we call other fixtures.
def modular_lib_path() -> Path:
    return Path(os.environ["MODULAR_PATH"]) / ".derived" / "build" / "lib"


@pytest.fixture
def sdk_test_inputs_path(modular_path: Path) -> Path:
    return modular_path / "SDK" / "integration-test" / "API" / "Inputs"


@pytest.fixture
def relu_torchscript_model_path(sdk_test_inputs_path: Path) -> Path:
    return sdk_test_inputs_path / "relu3x100x100.torchscript"


@pytest.fixture
def custom_ops_package_path(request) -> Path:
    return Path(
        os.getenv("CUSTOM_OPS_PATH")
        or request.config.getoption("--custom-ops-path")
    ).absolute()


@pytest.fixture
def mo_listio_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated model with list I/O."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "mo-list-model.mlir"
    )


def test_execute_success(
    session: InferenceSession, mo_model_path: Path
) -> None:
    model = session.load(mo_model_path)
    output = model.execute(np.ones(5, dtype=np.float32))
    assert len(output) == 1
    assert isinstance(output[0], Tensor)
    assert np.allclose(
        output[0].to_numpy(),
        np.array([4.0, 2.0, -5.0, 3.0, 6.0], dtype=np.float32),
    )


def test_devicetensor_wrong_num_inputs(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing with the
    # wrong number of input tensors.
    model = session.load(mo_model_path)
    first_tensor = Tensor(DType.float32, (5,))
    second_tensor = Tensor(DType.float32, (5,))
    # Ensure that tensors are initialized
    for i in range(5):
        first_tensor[i] = i
        second_tensor[i] = i
    with pytest.raises(
        ValueError,
        match=(
            r"Number of inputs does not match "
            r"expected number \(1\) for model"
        ),
    ):
        model.execute(first_tensor, second_tensor)


def test_devicetensor_wrong_shape(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing a tensor with
    # the wrong shape.
    model = session.load(mo_model_path)
    tensor = Tensor(DType.float32, (6,))
    # Ensure that tensors are initialized
    for i in range(6):
        tensor[i] = i
    with pytest.raises(
        ValueError,
        match=(
            r"Shape mismatch at position 0: expected tensor dimension "
            r"to be 5 at axis 0 but found dimension to be 6 instead"
        ),
    ):
        model.execute(tensor)


def test_devicetensor_wrong_rank(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing a tensor with
    # the wrong shape.
    model = session.load(mo_model_path)
    tensor = Tensor(DType.float32, (5, 2))
    # Ensure that tensors are initialized
    for i in range(5):
        for j in range(2):
            tensor[i, j] = i
    with pytest.raises(
        ValueError,
        match=(
            r"Rank mismatch: expected a tensor of rank 1 at position 0 "
            r"but got a tensor of rank 2 instead."
        ),
    ):
        model.execute(tensor)


def test_devicetensor_wrong_dtype(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing a tensor with
    # the wrong dtype.
    model = session.load(mo_model_path)
    tensor = Tensor(DType.int32, (6,))
    # Ensure that tensors are initialized
    for i in range(6):
        tensor[i] = i
    with pytest.raises(
        ValueError,
        match=(
            r"DType mismatch: expected f32 at position 0 but got si32 instead."
        ),
    ):
        model.execute(tensor)


def test_execute_device_tensor(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should be able to take in a simple 1-d tensor and execute a
    # model with this input.
    model = session.load(mo_model_path)
    input_tensor = Tensor(DType.float32, (5,))
    for idx in range(5):
        input_tensor[idx] = 1.0
    output = model.execute(input_tensor)
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Tensor)
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])


def test_execute_noncontiguous_tensor(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should reject any strided tensor inputs and request that they
    # be reallocated using `.contiguous`.
    model = session.load(mo_model_path)
    input_tensor = Tensor(DType.float32, (10,))
    for idx in range(10):
        input_tensor[idx] = 1.0
    subtensor = input_tensor[::2]
    with pytest.raises(
        ValueError,
        match=(
            r"Max does not currently support executing "
            r"non-contiguous tensors."
        ),
    ):
        model.execute(subtensor)
    output = model.execute(subtensor.contiguous())
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Tensor)
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])


def test_execute_devicetensor_dynamic_shape(
    session: InferenceSession, dynamic_model_path: Path
) -> None:
    # Device tensors should be able to execute even when the model expects
    # dynamic shapes.
    model = session.load(dynamic_model_path)
    tensor_one = Tensor(DType.int32, (5,))
    tensor_two = Tensor(DType.int32, (5,))

    for x in range(5):
        tensor_one[x] = x
        tensor_two[x] = 2 * x

    outputs = model.execute(tensor_one, tensor_two)
    assert len(outputs) == 1
    output_tensor = outputs[0]
    assert isinstance(output_tensor, Tensor)
    for x in range(5):
        assert output_tensor[x].item() == 3 * x


def test_execute_devicetensor_numpy_stays_alive(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # Our engine takes ownership of inputs and readily destroys them
    # after execution is complete. We need to ensure that when we create
    # a tensor from a numpy array, the original numpy array stays alive
    # after execution.
    model = session.load(mo_model_path)
    arr = np.ones((5,), dtype=np.float32)
    input_tensor = Tensor.from_numpy(arr)
    output = model.execute(input_tensor)
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Tensor)
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])

    for idx in range(5):
        assert isclose(arr[idx].item(), 1.0)


def test_execute_subtensor(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # Our engine should be able to execute tensors that are contiguous slices
    # of larger tensors. This will be important for things like our kv cache
    # implementation.
    model = session.load(mo_model_path)
    arr = np.arange(0, 20, dtype=np.float32).reshape((2, 10))
    input_tensor = Tensor.from_numpy(arr)
    output = model.execute(input_tensor[0, :5])
    expected = [3.0, 2.0, -4.0, 5.0, 9.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Tensor)
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])

    # Let's ensure that execution doesn't delete the underlying numpy array.
    np.array_equal(arr, np.ones((2, 10), dtype=np.float32))

    # We need to also handle situations where we're creating tensors from numpy
    # arrays that have already been sliced.
    presliced_input = Tensor.from_numpy(arr[0, ::2])
    presliced_output = model.execute(presliced_input)
    presliced_expected = [3.0, 3.0, -2.0, 8.0, 13.0]
    assert len(presliced_output) == 1
    presliced_output_tensor = presliced_output[0]
    assert isinstance(presliced_output_tensor, Tensor)
    for idx in range(5):
        assert isclose(
            presliced_output_tensor[idx].item(),
            presliced_expected[idx],
        )


def test_no_devicetensor_inputs(
    session: InferenceSession, no_input_path: Path
) -> None:
    # The device tensor execution path should support models that take in no
    # input tensors.
    model = session.load(no_input_path)
    outputs = model.execute()
    assert len(outputs) == 1
    tensor_output = outputs[0]
    assert isinstance(tensor_output, Tensor)
    output = tensor_output.to_numpy()
    expected = np.arange(1, 6, dtype=np.int32)
    assert np.array_equal(output, expected)


def test_scalar_inputs(
    session: InferenceSession, scalar_input_path: Path
) -> None:
    # We should be able to execute models with scalar inputs.
    model = session.load(scalar_input_path)
    scalar = Tensor.scalar(3, dtype=DType.int32)
    vector = np.arange(1, 6, dtype=np.int32)

    output = model.execute(scalar, vector)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))

    # We should also be able to execute with raw Python scalars.
    output = model.execute(3, vector)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))

    # We should also be able to execute with numpy scalars.
    output = model.execute(np.int32(3), vector)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))


def test_numpy_aliasing() -> None:
    # dlpack expects that we alias in this situation
    # https://dmlc.github.io/dlpack/latest/python_spec.html#semantics
    tensor = Tensor.zeros((5,), DType.int32, device=CPU())
    tensor_numpy = tensor.to_numpy()

    tensor[0] = 5
    assert tensor_numpy[0] == 5


def test_aliasing_output(
    session: InferenceSession, aliasing_outputs_path: Path
) -> None:
    # The device tensor execution path should support models that return the
    # same tensor outputs more than once.
    model = session.load(aliasing_outputs_path)
    arr = np.arange(0, 5, dtype=np.int32)
    input_tensor = Tensor.from_numpy(arr)
    outputs = model.execute(input_tensor)
    assert len(outputs) == 2
    x_tensor, y_tensor = outputs

    expected = np.arange(0, 10, 2, dtype=np.int32)

    assert isinstance(x_tensor, Tensor)
    x_numpy = x_tensor.to_numpy()
    assert np.array_equal(x_numpy, expected)

    assert isinstance(y_tensor, Tensor)
    y_numpy = y_tensor.to_numpy()
    assert np.array_equal(y_numpy, expected)

    # Check if the outputs really alias.
    x_tensor[0] = 7
    assert y_tensor[0].item() == 7


def test_list_io(session: InferenceSession, mo_listio_model_path: Path) -> None:
    model_with_list_io = session.load(mo_listio_model_path)
    output = model_with_list_io.execute_legacy(
        input_list=[np.zeros(2)], input_tensor=np.ones(5)
    )
    assert "output_list" in output
    output_list = output["output_list"]
    assert len(output_list) == 3
    assert np.allclose(output_list[0], np.zeros(2))
    assert np.allclose(output_list[1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(output_list[2], np.ones(5))


def test_dynamic_rank_spec() -> None:
    input_spec = TensorSpec(None, DType.float64, "dynamic")
    assert input_spec.shape is None
    assert input_spec.dtype == DType.float64
    assert input_spec.name == "dynamic"

    assert (
        repr(input_spec)
        == "TensorSpec(shape=None, dtype=DType.float64, name=dynamic)"
    )
    assert str(input_spec) == "None x float64"


def test_repr_torch_input_spec() -> None:
    input_spec_with_shape = TorchInputSpec([20, 30], DType.float32)
    assert input_spec_with_shape.shape == [20, 30]
    assert input_spec_with_shape.dtype == DType.float32

    assert (
        repr(input_spec_with_shape)
        == "TorchInputSpec(shape=[20, 30], dtype=DType.float32, device='')"
    )
    assert str(input_spec_with_shape) == "20x30xfloat32"

    input_spec_without_shape = TorchInputSpec(None, DType.float64)
    assert input_spec_without_shape.shape is None
    assert input_spec_without_shape.dtype == DType.float64

    assert (
        repr(input_spec_without_shape)
        == "TorchInputSpec(shape=None, dtype=DType.float64, device='')"
    )
    assert str(input_spec_without_shape) == "None x float64"

    input_spec_with_dim_names = TorchInputSpec(["BATCH", 30], DType.float32)
    assert input_spec_with_dim_names.shape == ["BATCH", 30]
    assert (
        repr(input_spec_with_dim_names)
        == "TorchInputSpec(shape=['BATCH', 30], dtype=DType.float32, device='')"
    )
    assert str(input_spec_with_dim_names) == "-1x30xfloat32"


@dataclass
class ExternalWeightsModel:
    num_elems: int

    def __call__(self, input: Value) -> Value:
        weights_tensor_type = TensorType(
            DType.float32, (self.num_elems,)
        ).to_mlir()
        weights_tensor = Graph.current._add_op(
            mo.constant_external,
            result=weights_tensor_type,
            name="foo",
            align=np.dtype(np.float32).alignment,
        )[0]

        # Set the constant external op's device explicitly.
        const_external_op = weights_tensor._mlir_value.owner
        const_external_op.attributes["device"] = DeviceRef.CPU().to_mlir()

        return input + weights_tensor


@pytest.fixture(scope="module")
def external_weights_size() -> int:
    return 4096


@pytest.fixture(scope="module")
def external_weights_graph(external_weights_size: int) -> Graph:
    graph = Graph(
        "external_weights",
        ExternalWeightsModel(external_weights_size),
        input_types=(TensorType(DType.float32, (external_weights_size,)),),
    )
    graph._mlir_op.verify()
    return graph


def test_execute_external_weights_numpy(
    session: InferenceSession,
    external_weights_graph: Graph,
    external_weights_size: int,
) -> None:
    weights = np.arange(external_weights_size, dtype=np.float32)
    compiled = session.load(
        external_weights_graph, weights_registry={"foo": weights}
    )

    input = np.random.randn(external_weights_size).astype(np.float32)
    output = compiled.execute(input)
    assert isinstance(output[0], Tensor)
    assert np.allclose(output[0].to_numpy(), input + weights)


def test_execute_external_weights_torch(
    session: InferenceSession,
    external_weights_graph: Graph,
    external_weights_size: int,
) -> None:
    weights = torch.arange(external_weights_size, dtype=torch.float32)
    compiled = session.load(
        external_weights_graph, weights_registry={"foo": weights}
    )

    input = torch.randn(external_weights_size, dtype=torch.float32)
    output = compiled.execute(input)
    assert torch.allclose(torch.from_dlpack(output[0]), input + weights)


def test_stats_report(
    session: InferenceSession, relu_torchscript_model_path: Path
) -> None:
    input_specs = [TorchInputSpec(shape=[1, 3, 100, 100], dtype=DType.float32)]
    session.load(relu_torchscript_model_path, input_specs=input_specs)
    sr = session.stats_report
    assert isinstance(sr, dict)
    assert sr["fallbacks"] == {}
    assert sr["total_op_count"] == 3


def test_devices(session: InferenceSession) -> None:
    host = CPU()
    assert str(host) == str(session.devices[0])


@pytest.fixture
def call_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Fixture for inputs to __call__ tests.
    a = Tensor.from_numpy(np.arange(0, 5, dtype=np.int32))
    b = Tensor.from_numpy(np.arange(5, 10, dtype=np.int32))
    c = Tensor.from_numpy(np.arange(10, 15, dtype=np.int32))
    d = Tensor.from_numpy(np.arange(15, 20, dtype=np.int32))
    e = Tensor.from_numpy(np.arange(20, 25, dtype=np.int32))
    return (a, b, c, d, e)


@pytest.fixture
def call_output() -> np.ndarray:
    # Expected output for __call__ tests.
    return np.array([50, 55, 60, 65, 70], dtype=np.int32)


@pytest.fixture
def call_model(session: InferenceSession, named_inputs_path: Path) -> Model:
    # Loaded model for __call__ tests.
    return session.load(named_inputs_path)


def test_positional_call(
    call_inputs: tuple, call_output: np.ndarray, call_model: Model
) -> None:
    # Calling a model with strictly positional inputs should work.
    a, b, c, d, e = call_inputs
    output = call_model(a, b, c, d, e)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(call_output, output.to_numpy())


def test_named_call(
    call_inputs: tuple, call_output: np.ndarray, call_model: Model
) -> None:
    # Calling a model with strictly named inputs should work.
    a, b, c, d, e = call_inputs
    output = call_model(b=b, a=a, e=e, c=c, d=d)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(call_output, output.to_numpy())


def test_mixed_positional_named_call(
    call_inputs: tuple, call_output: np.ndarray, call_model: Model
) -> None:
    # Calling a model with a mixture of named and positional inputs should also work (even if named
    # inputs are not ordered).
    a, b, c, d, e = call_inputs
    output = call_model(a, b, e=e, c=c, d=d)[0]
    assert isinstance(output, Tensor)
    assert np.array_equal(call_output, output.to_numpy())


def test_too_few_inputs_call(call_inputs: tuple, call_model: Model) -> None:
    # Calling a model with less inputs than expected should not work.
    a, b, c, _, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, e=e, c=c)


def test_too_many_inputs_call(call_inputs: tuple, call_model: Model) -> None:
    # Calling a model with more inputs than expected should not work.
    a, b, c, d, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, c, d, e, a)


def test_already_specified_input_call(
    call_inputs: tuple, call_model: Model
) -> None:
    # Calling a model with inputs that correspond to indexes already occupied by
    # positional inputs should not work.
    a, b, c, d, _ = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, b=b, c=c, d=d)


def test_unrecognized_name_call(call_inputs: tuple, call_model: Model) -> None:
    # Calling model with unrecognized names should not work.
    a, b, c, d, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, f=e, c=c, d=d)


def test_invalid_session_arg() -> None:
    """Check that passing an invalid arg to InferenceSession's ctor errors."""
    with pytest.raises(TypeError):
        InferenceSession(device=[])  # type: ignore
