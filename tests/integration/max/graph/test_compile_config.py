# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max.dtype import DType
from max.engine import InferenceSession, LogLevel
from max.graph import Graph, TensorType, ops


@pytest.fixture
def compile_config_ops_path() -> Path:
    return Path(os.environ["MODULAR_COMPILE_CONFIG_OPS_PATH"])


def test_compile_config_split_k_reduction_scheme(
    session: InferenceSession, compile_config_ops_path: Path
):
    tensor_type = TensorType(dtype=DType.int32, shape=[1])
    with Graph("graph", input_types=[]) as graph:
        graph.output(
            ops.custom("use_splitk_reduction_scheme", [], [tensor_type])[0]
        )

    session.set_split_k_reduction_precision("ACCUM")
    model = session.load(graph, custom_extensions=compile_config_ops_path)
    result = model.execute()[0].to_numpy()
    assert result == [1]

    session.set_split_k_reduction_precision("OUTPUT")
    model = session.load(graph, custom_extensions=compile_config_ops_path)
    result = model.execute()[0].to_numpy()
    assert result == [2]


def test_compile_config_use_logger(
    capfd, session: InferenceSession, compile_config_ops_path: Path
):
    tensor_type = TensorType(dtype=DType.int32, shape=[1])
    with Graph("graph", input_types=[]) as graph:
        graph.output(ops.custom("use_logger", [], [tensor_type])[0])

    session.set_mojo_log_level(LogLevel.DEBUG)
    model = session.load(graph, custom_extensions=compile_config_ops_path)
    result = model.execute()[0].to_numpy()

    # On the Mojo side the logger level is set to DEBUG, so the result should be 10.
    assert result == [10]

    # On the Mojo side, the logger has printed "I'm a custom Mojo function!"
    # We need to capture the output and check that it matches.
    captured = capfd.readouterr()
    assert captured.out == "ERROR::: I'm a custom Mojo function!\n"
