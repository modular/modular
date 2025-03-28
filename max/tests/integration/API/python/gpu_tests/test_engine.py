# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Value
from max.mlir.dialects import mo


def test_load_on_gpu(gpu_session: InferenceSession, mo_model_path: Path):
    """Verify we can compile and load a model on GPU."""
    _ = gpu_session.load(mo_model_path)


def test_execute_gpu(gpu_session: InferenceSession, mo_model_path: Path):
    """Validate that we can execute inputs on GPU."""
    model = gpu_session.load(mo_model_path)
    cuda = model.devices[0]
    input_tensor = Tensor.from_numpy(np.ones(5, dtype=np.float32)).to(cuda)
    outputs = model.execute(input_tensor)
    assert len(outputs) == 1
    output_tensor = outputs[0]
    host_tensor = output_tensor.to(CPU())
    for idx, elt in enumerate([4.0, 2.0, -5.0, 3.0, 6.0]):
        assert isclose(host_tensor[idx].item(), elt)


def test_execute_subtensor(gpu_session: InferenceSession, mo_model_path: Path):
    # Our engine should be able to execute tensors that are contiguous slices
    # of larger tensors. This will be important for things like our kv cache
    # implementation.
    model = gpu_session.load(mo_model_path)
    cuda = model.devices[0]

    arr = np.arange(0, 20, dtype=np.float32).reshape((2, 10))
    input_tensor = Tensor.from_numpy(arr).to(cuda)[0, :5]
    outputs = model.execute(input_tensor)
    assert len(outputs) == 1
    output_tensor = outputs[0]
    assert not output_tensor.is_host
    host_tensor = output_tensor.to(CPU())
    expected = [3.0, 2.0, -4.0, 5.0, 9.0]
    for idx, elt in enumerate(expected):
        assert isclose(host_tensor[idx].item(), elt)

    # Let's ensure that execution doesn't delete the underlying numpy array.
    np.array_equal(arr, np.ones((2, 10), dtype=np.float32))

    # We need to also handle situations where we're creating tensors from numpy
    # arrays that have already been sliced.
    presliced_input = Tensor.from_numpy(arr[0, ::2]).to(cuda)
    presliced_output = model.execute(presliced_input)
    presliced_expected = [3.0, 3.0, -2.0, 8.0, 13.0]
    assert len(presliced_output) == 1
    presliced_output_tensor_host = presliced_output[0].to(CPU())
    for idx in range(5):
        assert isclose(
            presliced_output_tensor_host[idx].item(), presliced_expected[idx]
        )


def test_scalar_inputs(gpu_session: InferenceSession, scalar_input_path: Path):
    # We should be able to execute models with scalar inputs.
    model = gpu_session.load(scalar_input_path)
    cuda = model.devices[0]
    scalar = Tensor.scalar(3, dtype=DType.int32, device=cuda)
    vector = np.arange(1, 6, dtype=np.int32)

    cuda_output = model.execute(scalar, vector)[0]
    host_output = cuda_output.to(CPU())
    assert np.array_equal(
        host_output.to_numpy(), np.arange(4, 9, dtype=np.int32)
    )

    # We should also be able to execute with raw Python scalars.
    cuda_output = model.execute(3, vector)[0]
    host_output = cuda_output.to(CPU())
    assert np.array_equal(
        host_output.to_numpy(), np.arange(4, 9, dtype=np.int32)
    )

    # We should also be able to execute with numpy scalars.
    cuda_output = model.execute(np.int32(3), vector)[0]
    host_output = cuda_output.to(CPU())
    assert np.array_equal(
        host_output.to_numpy(), np.arange(4, 9, dtype=np.int32)
    )


@dataclass
class Model:
    """Model that performs elementwise add with a weights tensor."""

    num_elems: int
    device: DeviceRef = field(default_factory=CPU)

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
        const_external_op.attributes["device"] = (
            DeviceRef.CPU().to_mlir()
            if self.device.is_host
            else DeviceRef.GPU(0).to_mlir()
        )

        return input + weights_tensor


def test_execute_external_weights_gpu(gpu_session: InferenceSession) -> None:
    num_elems = 4096
    weights = np.arange(num_elems, dtype=np.float32)

    graph = Graph(
        "external_weights",
        Model(num_elems),
        input_types=(TensorType(DType.float32, (num_elems,)),),
    )

    compiled = gpu_session.load(graph, weights_registry={"foo": weights})
    cuda = compiled.devices[0]
    input_np = (
        np.random.default_rng(seed=42)
        .standard_normal(num_elems)
        .astype(np.float32)
    )
    output = compiled.execute(
        Tensor.from_dlpack(input_np).to(cuda),
        copy_inputs_to_device=False,
    )[0].to(CPU())
    for idx, elt in enumerate(input_np + weights):
        assert isclose(output[idx].item(), elt)


def test_execute_external_weights_gpu_resident() -> None:
    """Executes a model with external weights already resident on device."""
    cuda = Accelerator()
    gpu_session = InferenceSession(devices=[cuda])

    num_elems = 4096
    weights_np = np.arange(num_elems, dtype=np.float32)
    weights = Tensor.from_dlpack(weights_np).to(cuda)

    graph = Graph(
        "external_weights_gpu_resident",
        Model(num_elems, device=cuda),
        input_types=(TensorType(DType.float32, (num_elems,)),),
    )

    # Check that this graph has a Accelerator constant external op.
    const_external_op = next(
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if isinstance(op, mo.ConstantExternalOp)
    )
    assert "gpu" in str(const_external_op.attributes["device"])

    # Compile and execute with the gpu-resident weights.
    compiled = gpu_session.load(graph, weights_registry={"foo": weights})

    input_np = (
        np.random.default_rng(seed=42)
        .standard_normal(num_elems)
        .astype(np.float32)
    )
    output = compiled.execute(Tensor.from_dlpack(input_np).to(cuda))[0].to(
        CPU()
    )

    # Check that the result is as expected.
    for idx, elt in enumerate(input_np + weights.to_numpy()):
        assert isclose(output[idx].item(), elt)


def test_no_devicetensor_inputs(
    gpu_session: InferenceSession, no_input_path: Path
):
    # The device tensor execution path should support models that take in no
    # input tensors.
    model = gpu_session.load(no_input_path)
    outputs = model.execute()
    assert len(outputs) == 1
    host_tensor = outputs[0].to(CPU())
    output = np.from_dlpack(host_tensor)
    expected = np.arange(1, 6, dtype=np.int32)
    assert np.array_equal(output, expected)


def test_aliasing_outputs(
    gpu_session: InferenceSession, aliasing_outputs_path: Path
):
    # The device tensor execution path should support models that return the
    # same tensor outputs more than once.
    model = gpu_session.load(aliasing_outputs_path)
    cuda = model.devices[0]

    arr = np.arange(0, 5, dtype=np.int32)
    input_tensor = Tensor.from_numpy(arr).to(cuda)
    outputs = model.execute(input_tensor)
    assert len(outputs) == 2

    tensor_output0 = outputs[0].to(CPU())
    array_output0 = tensor_output0.to_numpy()
    expected = np.arange(0, 10, 2, dtype=np.int32)
    assert np.array_equal(array_output0, expected)

    tensor_output1 = outputs[1].to(CPU())
    array_output1 = tensor_output1.to_numpy()
    assert np.array_equal(array_output1, expected)

    # Check if the outputs really alias.
    # TODO: enable this when we have GPU indexing.
    # tensor_output0[0] = 7
    # assert array_output1[0] == 7


def test_devices(gpu_session: InferenceSession) -> None:
    device = Accelerator()
    assert str(device) == str(gpu_session.devices[0])
