# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import isclose
from pathlib import Path

import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.engine import InferenceSession


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
