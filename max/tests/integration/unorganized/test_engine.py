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
"""Test the max.engine Python bindings with MOF."""

import os
import re
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    Weight,
)
from max.mlir.dialects import mo

# This path is used in skipif clauses rather than tests, so we can neither mark

# it as a fixture nor can we call other fixtures


def modular_lib_path() -> Path:
    return Path(os.environ["MODULAR_PATH"]) / ".derived/build/lib"


@pytest.fixture
def custom_ops_package_path(request: pytest.FixtureRequest) -> Path:
    return Path(
        os.getenv("CUSTOM_OPS_PATH")
        or request.config.getoption("--custom-ops-path")
    ).absolute()


@pytest.fixture
def mo_listio_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated model with list I/O."""
    return modular_path / "max/tests/integration/API/Inputs/mo-list-model.mlir"


def test_execute_success(
    session: InferenceSession, mo_model_path: Path
) -> None:
    model = session.load(mo_model_path)
    output = model.execute(
        Buffer.from_numpy(np.ones(5, dtype=np.float32)).to(
            model.input_devices[0]
        )
    )
    assert len(output) == 1
    assert isinstance(output[0], Buffer)
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
    first_tensor = Buffer(DType.float32, (5,))
    second_tensor = Buffer(DType.float32, (5,))
    # Ensure that tensors are initialized
    for i in range(5):
        first_tensor[i] = i
        second_tensor[i] = i

    first_tensor = first_tensor.to(model.input_devices[0])
    second_tensor = second_tensor.to(model.input_devices[0])
    with pytest.raises(
        ValueError,
        match=(
            r"Number of inputs \(2\) does not match "
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
    tensor = Buffer(DType.float32, (6,))
    # Ensure that tensors are initialized
    for i in range(6):
        tensor[i] = i

    tensor = tensor.to(model.input_devices[0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input at position 0: Buffer of type [(6), f32] does not match expected type [(5), f32]",
        ),
    ):
        model.execute(tensor)


def test_devicetensor_wrong_rank(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing a tensor with
    # the wrong shape.
    model = session.load(mo_model_path)
    tensor = Buffer(DType.float32, (5, 2))
    # Ensure that tensors are initialized
    for i in range(5):
        for j in range(2):
            tensor[i, j] = i

    tensor = tensor.to(model.input_devices[0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input at position 0: Buffer of type [(5, 2), f32] does not match expected type [(5), f32]"
        ),
    ):
        model.execute(tensor)


def test_devicetensor_wrong_dtype(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should throw a ValueError when executing a tensor with
    # the wrong dtype.
    model = session.load(mo_model_path)
    tensor = Buffer(DType.int32, (5,))
    # Ensure that tensors are initialized
    for i in range(5):
        tensor[i] = i

    tensor = tensor.to(model.input_devices[0])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input at position 0: Buffer of type [(5), si32] does not match expected type [(5), f32]"
        ),
    ):
        model.execute(tensor)


def test_execute_device_tensor(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should be able to take in a simple 1-d tensor and execute a
    # model with this input.
    model = session.load(mo_model_path)
    input_tensor = Buffer(DType.float32, (5,))
    for idx in range(5):
        input_tensor[idx] = 1.0

    input_tensor = input_tensor.to(model.input_devices[0])
    output = model.execute(input_tensor)
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Buffer)
    output_tensor = output_tensor.to(CPU())
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])


def test_execute_noncontiguous_tensor(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # The engine should reject any strided tensor inputs and request that they
    # be reallocated using `.contiguous`.
    model = session.load(mo_model_path)
    input_tensor = Buffer(DType.float32, (10,))
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

    device = CPU()
    output = model.execute(subtensor.contiguous().to(device))
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Buffer)
    output_tensor = output_tensor.to(CPU())
    for idx in range(5):
        assert isclose(output_tensor[idx].item(), expected[idx])


def test_execute_devicetensor_numpy_stays_alive(
    session: InferenceSession, mo_model_path: Path
) -> None:
    # Our engine takes ownership of inputs and readily destroys them
    # after execution is complete. We need to ensure that when we create
    # a tensor from a numpy array, the original numpy array stays alive
    # after execution.
    model = session.load(mo_model_path)
    arr = np.ones((5,), dtype=np.float32)
    input_tensor = Buffer.from_numpy(arr).to(model.input_devices[0])
    output = model.execute(input_tensor)
    expected = [4.0, 2.0, -5.0, 3.0, 6.0]
    assert len(output) == 1
    output_tensor = output[0]
    assert isinstance(output_tensor, Buffer)
    output_tensor = output_tensor.to(CPU())
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
    input_tensor = Buffer.from_numpy(arr).to(CPU())
    outputs = model.execute(input_tensor[0, :5])
    assert len(outputs) == 1
    output_tensor = outputs[0]
    assert isinstance(output_tensor, Buffer)
    host_tensor = output_tensor.to(CPU())
    expected = [3.0, 2.0, -4.0, 5.0, 9.0]
    for idx, elt in enumerate(expected):
        assert isclose(host_tensor[idx].item(), elt)

    # Let's ensure that execution doesn't delete the underlying numpy array.
    np.array_equal(arr, np.ones((2, 10), dtype=np.float32))

    # We need to also handle situations where we're creating tensors from numpy
    # arrays that have already been sliced.
    presliced_input = Buffer.from_numpy(arr[0, ::2]).to(CPU())
    presliced_output = model.execute(presliced_input)
    presliced_expected = [3.0, 3.0, -2.0, 8.0, 13.0]
    assert len(presliced_output) == 1
    assert isinstance(presliced_output[0], Buffer)
    presliced_output_tensor_host = presliced_output[0].to(CPU())
    for idx in range(5):
        assert isclose(
            presliced_output_tensor_host[idx].item(), presliced_expected[idx]
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
    assert isinstance(tensor_output, Buffer)
    output = tensor_output.to_numpy()
    expected = np.arange(1, 6, dtype=np.int32)
    assert np.array_equal(output, expected)


def test_scalar_inputs(
    session: InferenceSession, scalar_input_path: Path
) -> None:
    # We should be able to execute models with scalar inputs.
    model = session.load(scalar_input_path)
    scalar = Buffer.scalar(3, dtype=DType.int32).to(CPU())
    vector = Buffer.from_numpy(np.arange(1, 6, dtype=np.int32)).to(CPU())

    output = model.execute(scalar, vector)[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))

    # We should also be able to execute with raw Python scalars.
    output = model.execute(3, vector)[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))

    # We should also be able to execute with numpy scalars.
    output = model.execute(int(np.int32(3)), vector)[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(output.to_numpy(), np.arange(4, 9, dtype=np.int32))


def test_numpy_aliasing() -> None:
    # dlpack expects that we alias in this situation
    # https://dmlc.github.io/dlpack/latest/python_spec.html#semantics
    tensor = Buffer.zeros((5,), DType.int32, device=CPU())
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

    device = CPU()
    input_tensor = Buffer.from_numpy(arr).to(device)
    outputs = model.execute(input_tensor)
    assert len(outputs) == 2
    x_tensor, y_tensor = outputs

    expected = np.arange(0, 10, 2, dtype=np.int32)

    assert isinstance(x_tensor, Buffer)
    x_numpy = x_tensor.to(CPU()).to_numpy()
    assert np.array_equal(x_numpy, expected)

    assert isinstance(y_tensor, Buffer)
    y_numpy = y_tensor.to(CPU()).to_numpy()
    assert np.array_equal(y_numpy, expected)

    # Check if the outputs really alias.
    if accelerator_count() == 0:
        assert (
            x_tensor.to_numpy().ctypes.data == y_tensor.to_numpy().ctypes.data
        )


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="List is a CPU concept. Validating input and output lists is not supported on GPU.",
)
def test_list_io(session: InferenceSession, mo_listio_model_path: Path) -> None:
    model_with_list_io = session.load(mo_listio_model_path)
    # Convert to positional arguments using the model signature
    input_list = [np.zeros(2)]
    input_tensor = np.ones(5)
    # Cast to Any to satisfy mypy - the engine handles list inputs correctly at runtime
    output = model_with_list_io(cast(Any, input_list), input_tensor)

    # The new API returns a list, so we need to access by index instead of key
    assert len(output) == 1  # Assuming single output containing the list
    output_list = output[0]  # Get the first (and likely only) output
    # Type assertions to help mypy understand the types
    from typing import Any

    # Cast to Any to avoid mypy issues with union types
    output_list_any: Any = output_list
    assert len(output_list_any) == 3
    assert np.allclose(output_list_any[0], np.zeros(2))
    assert np.allclose(output_list_any[1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(output_list_any[2], np.ones(5))


@dataclass
class ExternalWeightsModel:
    """Model that performs elementwise add with a weights tensor."""

    num_elems: int
    device: Device = field(default_factory=CPU)

    def __call__(self, input: TensorValueLike) -> Value:
        weights_tensor = Weight(
            "foo",
            DType.float32,
            (self.num_elems,),
            DeviceRef.CPU() if self.device.is_host else DeviceRef.GPU(),
        )
        return TensorValue(input) + weights_tensor


@dataclass
class ExternalWeightsModelWithAlias:
    """Model that performs elementwise add with a weights tensor."""

    num_elems: int
    device: Device = field(default_factory=CPU)

    def __call__(self, input: TensorValueLike) -> Value:
        weights_tensor = Weight(
            "foo",
            DType.float32,
            (self.num_elems,),
            DeviceRef.CPU() if self.device.is_host else DeviceRef.GPU(),
            _has_alias=True,
        )
        return TensorValue(input) + weights_tensor


@pytest.fixture(scope="module")
def external_weights_size() -> int:
    return 4096


@pytest.fixture(scope="module")
def external_weights_graph(external_weights_size: int) -> Graph:
    graph = Graph(
        "external_weights",
        ExternalWeightsModel(external_weights_size),
        input_types=(
            TensorType(
                DType.float32, (external_weights_size,), DeviceRef.CPU()
            ),
        ),
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
    output = compiled(input)
    assert isinstance(output[0], Buffer)
    assert np.allclose(output[0].to_numpy(), input + weights)


@pytest.mark.skipif(
    accelerator_count() > 0,
    reason="This fails on gpu due to tensors on different devices, yet the numpy test above passes.",
)
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
    output = compiled.execute(
        Buffer.from_dlpack(input).to(compiled.input_devices[0])
    )
    assert torch.allclose(torch.from_dlpack(output[0]), input + weights)


def test_execute_external_weights_resident(
    session: InferenceSession,
    external_weights_size: int,
) -> None:
    """Executes a model with external weights already resident on device."""
    weights_np = np.arange(external_weights_size, dtype=np.float32)
    weights = Buffer.from_numpy(weights_np)

    graph = Graph(
        "external_weights_gpu_resident",
        ExternalWeightsModel(external_weights_size, device=session.devices[0]),
        input_types=(
            TensorType(
                DType.float32,
                (external_weights_size,),
                DeviceRef.from_device(session.devices[0]),
            ),
        ),
    )
    compiled = session.load(graph, weights_registry={"foo": weights})

    # Check that this graph has a device constant external op.
    const_external_op = next(
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if isinstance(op, mo.ConstantExternalOp)
    )

    # Compile and execute with the device-resident weights.
    compiled = session.load(graph, weights_registry={"foo": weights})

    input_np = (
        np.random.default_rng(seed=42)
        .standard_normal(external_weights_size)
        .astype(np.float32)
    )
    output = compiled.execute(
        Buffer.from_numpy(input_np).to(compiled.input_devices[0])
    )
    assert isinstance(output[0], Buffer)
    assert np.allclose(output[0].to_numpy(), input_np + weights_np)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_weight_device_mismatch(
    session: InferenceSession,
) -> None:
    cuda = Accelerator()
    # Create graph with CPU-based weight
    with Graph(
        "test_device_validation",
        input_types=[TensorType(DType.float32, (10, 10), DeviceRef.GPU())],
    ) as g:
        x = g.inputs[0].tensor
        # Note: this is actually a weight on CPU which is transferred to GPU by runtime
        weight = Weight("w", DType.float32, (10, 10), device=DeviceRef.GPU())
        y = x @ weight
        g.output(y)

    # Create GPU tensor that should be on CPU
    device_weight = torch.tensor(np.ones((10, 10), dtype=np.float32)).to("cuda")

    # Create test input on GPU
    input_tensor = Buffer.from_numpy(np.ones((10, 10), dtype=np.float32)).to(
        cuda
    )

    # This will load but set up invalid device configuration
    with pytest.raises(
        ValueError, match=r"Mismatch in device type for weight 'w'."
    ):
        model = session.load(
            g,
            weights_registry={"w": device_weight},
        )

        result = model.execute(input_tensor)[0]
        assert isinstance(result, Buffer)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_execute_external_weights_resident_with_alias(
    session: InferenceSession,
) -> None:
    """Executes a model with external weights already resident on device."""

    num_elems = 4096
    weights_np = np.arange(num_elems, dtype=np.float32)
    weights = Buffer.from_dlpack(weights_np).to(Accelerator())

    graph = Graph(
        "external_weights_resident_with_alias",
        ExternalWeightsModelWithAlias(num_elems, device=Accelerator()),
        input_types=(
            TensorType(DType.float32, (num_elems,), device=DeviceRef.GPU()),
        ),
    )

    # Check that this graph has a CUDA constant external op.
    const_external_op = next(
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if isinstance(op, mo.ConstantExternalOp)
    )
    assert "gpu" in str(const_external_op.attributes["device"])

    # Compile and execute with the gpu-resident weights.
    compiled = session.load(graph, weights_registry={"foo": weights})

    input_np = (
        np.random.default_rng(seed=42)
        .standard_normal(num_elems)
        .astype(np.float32)
    )

    output_tensor = compiled.execute(
        Buffer.from_dlpack(input_np).to(Accelerator())
    )[0]
    assert isinstance(output_tensor, Buffer)
    output = output_tensor.to(CPU()).to_numpy()

    # Check that the result is as expected.
    for idx, elt in enumerate(input_np + weights.to(CPU()).to_numpy()):
        assert isclose(output[idx].item(), elt)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_weight_device_implicit_mismatch(
    session: InferenceSession,
) -> None:
    cuda = Accelerator()
    with Graph(
        "test_device_validation",
        input_types=[
            TensorType(DType.float32, (10, 10), device=DeviceRef.GPU())
        ],
    ) as g:
        x = g.inputs[0].tensor

        # Note: weight is implicitly on host (and has automatic transfer inserted)
        weight = Weight("w", DType.float32, (10, 10), device=DeviceRef.GPU())
        y = x @ weight
        g.output(y)

    # Create GPU tensor that should be on CPU
    device_weight = torch.tensor(np.ones((10, 10), dtype=np.float32)).to("cuda")

    # Create test input on GPU
    input_tensor = Buffer.from_numpy(np.ones((10, 10), dtype=np.float32)).to(
        cuda
    )

    # This will load but set up invalid device configuration
    with pytest.raises(
        ValueError, match=r"Mismatch in device type for weight 'w'."
    ):
        model = session.load(
            g,
            weights_registry={"w": device_weight},
        )

        result = model.execute(input_tensor)[0]
        assert isinstance(result, Buffer)


def test_devices(session: InferenceSession) -> None:
    host = CPU()
    if accelerator_count() == 0:
        assert str(host) == str(session.devices[0])
    else:
        assert str(Accelerator()) == str(session.devices[0])


@pytest.fixture
def call_inputs() -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer]:
    # Fixture for inputs to __call__ tests.
    a = Buffer.from_numpy(np.arange(0, 5, dtype=np.int32))
    b = Buffer.from_numpy(np.arange(5, 10, dtype=np.int32))
    c = Buffer.from_numpy(np.arange(10, 15, dtype=np.int32))
    d = Buffer.from_numpy(np.arange(15, 20, dtype=np.int32))
    e = Buffer.from_numpy(np.arange(20, 25, dtype=np.int32))
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
    call_inputs: tuple,
    call_output: np.ndarray,
    call_model: Model,
) -> None:
    # Calling a model with strictly positional inputs should work.
    a, b, c, d, e = call_inputs
    output = call_model(
        a.to(call_model.input_devices[0]),
        b.to(call_model.input_devices[1]),
        c.to(call_model.input_devices[2]),
        d.to(call_model.input_devices[3]),
        e.to(call_model.input_devices[4]),
    )[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(call_output, output.to_numpy())


def test_named_call(
    call_inputs: tuple,
    call_output: np.ndarray,
    call_model: Model,
) -> None:
    # Calling a model with strictly named inputs should work.
    a, b, c, d, e = call_inputs
    output = call_model(
        a=a.to(call_model.input_devices[0]),
        b=b.to(call_model.input_devices[1]),
        c=c.to(call_model.input_devices[2]),
        d=d.to(call_model.input_devices[3]),
        e=e.to(call_model.input_devices[4]),
    )[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(call_output, output.to_numpy())


def test_mixed_positional_named_call(
    call_inputs: tuple,
    call_output: np.ndarray,
    call_model: Model,
) -> None:
    # Calling a model with a mixture of named and positional inputs should also work (even if named
    # inputs are not ordered).
    a, b, c, d, e = call_inputs
    output = call_model(
        a.to(call_model.input_devices[0]),
        b.to(call_model.input_devices[1]),
        c=c.to(call_model.input_devices[2]),
        d=d.to(call_model.input_devices[3]),
        e=e.to(call_model.input_devices[4]),
    )[0]
    assert isinstance(output, Buffer)
    assert np.array_equal(call_output, output.to_numpy())


def test_too_few_inputs_call(
    call_inputs: tuple,
    call_model: Model,
) -> None:
    # Calling a model with less inputs than expected should not work.
    a, b, c, _, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, e=e, c=c)


def test_too_many_inputs_call(
    call_inputs: tuple,
    call_model: Model,
) -> None:
    # Calling a model with more inputs than expected should not work.
    a, b, c, d, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, c, d, e, a)


def test_already_specified_input_call(
    call_inputs: tuple,
    call_model: Model,
) -> None:
    # Calling a model with inputs that correspond to indexes already occupied by
    # positional inputs should not work.
    a, b, c, d, _ = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, b=b, c=c, d=d)


def test_unrecognized_name_call(
    call_inputs: tuple,
    call_model: Model,
) -> None:
    # Calling model with unrecognized names should not work.
    a, b, c, d, e = call_inputs
    with pytest.raises(TypeError):
        call_model(a, b, f=e, c=c, d=d)


def test_invalid_session_arg() -> None:
    """Check that passing an invalid arg to InferenceSession's ctor errors."""
    with pytest.raises(TypeError):
        InferenceSession(device=[])  # type: ignore


def test_session_device_initialization() -> None:
    """Verify InferenceSession device list initialization behavior."""
    cpu = CPU()

    # Case: Only CPU specified.
    session1 = InferenceSession(devices=[cpu])
    assert set(session1.devices) == {cpu}, (
        "Devices with only CPU should result in just CPU"
    )

    # Case: Duplicate CPU specified.
    session2 = InferenceSession(devices=[cpu, cpu])
    assert set(session2.devices) == {cpu}, (
        "Devices with duplicate CPU should result in just CPU"
    )

    # Case: Empty list specified.
    session3 = InferenceSession(devices=[])
    assert not set(session3.devices), (
        "Devices with empty list should give the empty set"
    )

    if accelerator_count() == 0:
        return

    gpu = Accelerator()
    # Case: Only GPU specified.
    session4 = InferenceSession(devices=[gpu])
    assert set(session4.devices) == {gpu}, (
        "Devices with only GPU should result in GPU"
    )

    # Case: GPU and CPU specified.
    session5 = InferenceSession(devices=[gpu, cpu])
    assert set(session5.devices) == {gpu, cpu}, (
        "Devices with GPU and CPU should be unique"
    )

    # Case: Duplicate GPU specified.
    session6 = InferenceSession(devices=[gpu, gpu])
    assert set(session6.devices) == {gpu}, (
        "Devices with duplicate GPU should be unique"
    )


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_execute_wrong_device_input(session: InferenceSession) -> None:
    """Test that model.execute() raises TypeError for input tensors on the wrong device."""
    N = 42
    model = session.load(
        Graph(
            "mixed_devices",
            forward=lambda x, y: x.to(cast(DeviceRef, y.device)) + y,
            input_types=[
                # One tensor on host, the other on device.
                TensorType(DType.float32, shape=[N], device=DeviceRef.CPU()),
                TensorType(DType.float32, shape=[N], device=DeviceRef.GPU()),
            ],
        )
    )

    # Attempt to execute with the wrong device input should raise TypeError
    with pytest.raises(
        TypeError,
        # Escape because CPU and Accelerator's repr's include parentheses.
        match=re.escape(
            f"expected argument 0 to be on device {CPU()}, but was on device {Accelerator()}"
        ),
    ):
        x = Buffer(DType.float32, shape=(N,), device=Accelerator())
        model.execute(x, x)
