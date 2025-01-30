# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Weight


def test_gpu_resident_weight(
    gpu_session: InferenceSession, graph_testdata: Path
) -> None:
    """Tests adding a GPU-resident external weight to a graph."""
    tensor_host = np.arange(10, dtype=np.int32)
    tensor_gpu = Tensor.from_numpy(tensor_host).to(Accelerator())
    weight = Weight("a", DType.int32, tensor_gpu.shape)

    with Graph("graph_with_pt_weights") as graph:
        const_external = graph.add_weight(weight, device=DeviceRef.GPU())
        graph.output(const_external + 1)

    compiled = gpu_session.load(
        graph,
        weights_registry={
            weight.name: tensor_gpu,
        },
    )

    output = compiled.execute()[0]
    assert (output.to(CPU()).to_numpy() == (tensor_host + 1)).all()
