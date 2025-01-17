# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


@pytest.fixture
def compile_config_ops_path() -> Path:
    return Path(os.environ["MODULAR_COMPILE_CONFIG_OPS_PATH"])


def test_compile_config(
    session: InferenceSession, compile_config_ops_path: Path
):
    tensor_type = TensorType(dtype=DType.int32, shape=[1])
    with Graph("compile_config", input_types=[]) as graph:
        graph.output(ops.custom("expose_env", [], [tensor_type])[0])

    session.set_split_k_reduction_precision("ACCUM")
    model = session.load(graph, custom_extensions=compile_config_ops_path)
    result = model.execute()[0].to_numpy()
    assert result == [1]

    session.set_split_k_reduction_precision("OUTPUT")
    model = session.load(graph, custom_extensions=compile_config_ops_path)
    result = model.execute()[0].to_numpy()
    assert result == [2]
