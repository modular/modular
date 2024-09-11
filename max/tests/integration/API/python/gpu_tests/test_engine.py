# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from dataclasses import dataclass
from math import isclose
from pathlib import Path

import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, Value
from max.mlir.dialects import mo


def test_load_on_gpu(gpu_session: InferenceSession, mo_model_path: Path):
    """Verify we can compile and load a model on GPU."""
    _ = gpu_session.load(mo_model_path)


def test_execute_gpu(gpu_session: InferenceSession, mo_model_path: Path):
    """Validate that we can execute inputs on GPU."""
    model = gpu_session.load(mo_model_path)
    input_tensor = Tensor.from_numpy(np.ones(5, dtype=np.float32), CUDA())
    outputs = model.execute(input_tensor)
    assert len(outputs) == 1
    output_tensor = outputs[0]
    host_tensor = output_tensor.copy_to(CPU())
    for idx, elt in enumerate([4.0, 2.0, -5.0, 3.0, 6.0]):
        assert isclose(host_tensor[idx].item(), elt)


def test_execute_subtensor(gpu_session: InferenceSession, mo_model_path: Path):
    # Our engine should be able to execute tensors that are contiguous slices
    # of larger tensors. This will be important for things like our kv cache
    # implementation.
    model = gpu_session.load(mo_model_path)
    arr = np.arange(0, 20, dtype=np.float32).reshape((2, 10))
    input_tensor = Tensor.from_numpy(arr, CUDA())[0, :5]
    outputs = model.execute(input_tensor)
    assert len(outputs) == 1
    output_tensor = outputs[0]
    assert not output_tensor.is_host
    host_tensor = output_tensor.copy_to(CPU())
    expected = [3.0, 2.0, -4.0, 5.0, 9.0]
    for idx, elt in enumerate(expected):
        assert isclose(host_tensor[idx].item(), elt)

    # Let's ensure that execution doesn't delete the underlying numpy array.
    np.array_equal(arr, np.ones((2, 10), dtype=np.float32))

    # We need to also handle situations where we're creating tensors from numpy
    # arrays that have already been sliced.
    presliced_input = Tensor.from_numpy(arr[0, ::2], CUDA())
    presliced_output = model.execute(presliced_input)
    presliced_expected = [3.0, 3.0, -2.0, 8.0, 13.0]
    assert len(presliced_output) == 1
    presliced_output_tensor = presliced_output[0].copy_to(CPU())
    for idx in range(5):
        assert isclose(
            presliced_output_tensor[idx].item(), presliced_expected[idx]
        )


@dataclass
class Model:
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

        return input + weights_tensor


def test_execute_external_weights_gpu(gpu_session: InferenceSession) -> None:
    num_elems = 4096
    weights = np.arange(num_elems, dtype=np.float32)

    graph = Graph(
        "external_weights",
        Model(num_elems),
        input_types=(TensorType(DType.float32, (num_elems,)),),
    )
    graph._mlir_op.verify()

    compiled = gpu_session.load(graph, weights_registry={"foo": weights})
    input_np = np.random.randn(num_elems).astype(np.float32)
    output = compiled.execute(Tensor.from_numpy(input_np, device=CUDA()))[
        0
    ].copy_to(CPU())
    for idx, elt in enumerate(input_np + weights):
        assert isclose(output[idx].item(), elt)


def test_no_devicetensor_inputs(
    gpu_session: InferenceSession, no_input_path: Path
):
    # The device tensor execution path should support models that take in no
    # input tensors.
    model = gpu_session.load(no_input_path)
    # We have to do this in kinda a jank way atm to force this to go through the
    # device tensor path. This will be simplified once we deprecate the named
    # tensor API.
    outputs = model._impl.execute_device_tensors([])
    assert len(outputs) == 1
    tensor_output = Tensor._from_impl(outputs[0])
    host_tensor = tensor_output.copy_to(CPU())
    output = np.from_dlpack(host_tensor)
    expected = np.arange(1, 6, dtype=np.int32)
    assert np.array_equal(output, expected)
