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
from max.graph import Graph, Value, TensorType
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
