# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

import os
import tempfile

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_graph_telemetry() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name
    os.environ["MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH"] = filepath
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    session = InferenceSession(devices=[CPU()])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        compiled = session.load(graph)
        a = np.ones((1, 1)).astype(np.float32)
        b = np.ones((1, 1)).astype(np.float32)
        _ = compiled.execute(a, b)

    expected_line = "max.pipeline.name: add"
    with open(filepath) as file:
        lines = [line.strip() for line in file]
    assert expected_line in lines
