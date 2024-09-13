# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, ops
from max.graph.type import TensorType
from max.graph.type import _OpaqueType as OpaqueType


@pytest.fixture
def counter_ops_path() -> Path:
    return Path(os.environ["COUNTER_OPS_PATH"])


@pytest.fixture
def maker_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = OpaqueType("Counter")
    maker_graph = Graph("maker", input_types=[], output_types=[counter_type])
    with maker_graph:
        maker_graph.output(ops.custom("make_counter", [], [counter_type])[0])
    maker_compiled = session.load(
        maker_graph, custom_extensions=counter_ops_path
    )
    return maker_compiled


@pytest.fixture
def bumper_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = OpaqueType("Counter")
    bumper_graph = Graph("bumper", input_types=[counter_type], output_types=[])
    with bumper_graph:
        # TODO(MSDK-950): Avoid DCE in the graph compiler and remove return value.
        c = ops.custom(
            "bump_counter",
            [bumper_graph.inputs[0]],
            [TensorType(DType.bool, [1])],
        )
        bumper_graph.output(c[0])
    bumper_compiled = session.load(
        bumper_graph, custom_extensions=counter_ops_path
    )
    return bumper_compiled


@pytest.fixture
def reader_model(session: InferenceSession, counter_ops_path: Path) -> Model:
    counter_type = OpaqueType("Counter")
    reader_graph = Graph("reader", input_types=[counter_type], output_types=[])
    with reader_graph:
        c = ops.custom(
            "read_counter",
            [reader_graph.inputs[0]],
            [TensorType(DType.int32, [2])],
        )
        reader_graph.output(c[0])
    reader_compiled = session.load(
        reader_graph, custom_extensions=counter_ops_path
    )
    return reader_compiled


def test_opaque(
    maker_model: Model, bumper_model: Model, reader_model: Model
) -> None:
    counter = maker_model.execute()["output0"]
    for i in range(5):
        bumper_model.execute(input0=counter)
    result = reader_model.execute(input0=counter)["output0"]

    assert (result == [5, 15]).all()


def test_opaque_driver_constructor(
    session: InferenceSession, counter_ops_path: Path
) -> None:
    """Tests constructing a Counter using the driver API."""
    counter_type = OpaqueType("Counter")

    maker_graph = Graph(
        "maker",
        input_types=[TensorType(DType.int32, (2,))],
        output_types=[counter_type],
    )
    with maker_graph:
        maker_graph.output(
            ops.custom(
                "make_counter_from_tensor",
                values=[maker_graph.inputs[0]],
                out_types=[counter_type],
            )[0]
        )
    maker_compiled = session.load(
        maker_graph, custom_extensions=counter_ops_path
    )

    init = Tensor((2,), DType.int32)
    init[0] = 42
    init[1] = 37
    counter = maker_compiled.execute(init)
    assert counter and counter[0]
