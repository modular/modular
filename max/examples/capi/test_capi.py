# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Simple MAX Graph example that adds two vectors."""

import os

from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def build_graph() -> None:
    input_type = TensorType(
        dtype=DType.float32, shape=(8,), device=DeviceRef.CPU()
    )

    # We'll just do simple one-operation vector addition for our graph
    with Graph("vector_add", input_types=(input_type, input_type)) as graph:
        vector1, vector2 = graph.inputs[0].tensor, graph.inputs[1].tensor

        output = vector1 + vector2  # Same as ops.add()

        graph.output(output)

    # Cmopile the graph
    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)

    # Save the graph to a MEF file
    model._export_mef("graph.mef")


def test_capi() -> None:
    build_graph()

    path = os.environ["GRAPH_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_capi()
