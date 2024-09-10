# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest

from max.engine import InferenceSession
from max.graph import Graph, ops
from max.graph.type import _OpaqueType as OpaqueType


@pytest.fixture
def counter_ops_path() -> Path:
    return Path(os.environ["COUNTER_OPS_PATH"])


def test_opaque(counter_ops_path: Path) -> None:
    session = InferenceSession()
    counter_type = OpaqueType("Counter")

    maker_graph = Graph("maker", input_types=[], output_types=[counter_type])
    with maker_graph:
        maker_graph.output(
            ops.custom("make_counter", [], [counter_type.to_mlir()])[0]
        )
    maker_compiled = session.load(
        maker_graph, custom_extensions=counter_ops_path
    )

    bumper_graph = Graph("bumper", input_types=[counter_type], output_types=[])
    with bumper_graph:
        ops.custom("bump_counter", [bumper_graph.inputs[0]], [])
        bumper_graph.output()
    bumper_compiled = session.load(
        bumper_graph, custom_extensions=counter_ops_path
    )

    dropper_graph = Graph(
        "dropper", input_types=[counter_type], output_types=[]
    )
    with dropper_graph:
        ops.custom("drop_counter", [dropper_graph.inputs[0]], [])
        dropper_graph.output()
    dropper_compiled = session.load(
        dropper_graph, custom_extensions=counter_ops_path
    )

    counter = maker_compiled.execute()["output0"]
    for i in range(5):
        bumper_compiled.execute(input0=counter)
    dropper_compiled.execute(input0=counter)


if __name__ == "__main__":
    test_opaque(os.environ["COUNTER_OPS_PATH"])
