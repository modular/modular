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

"""Generate a simple vector add MEF file for C API testing."""

import sys

from max import engine
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_mef_path>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]

    input_type = TensorType(
        dtype=DType.float32, shape=(8,), device=DeviceRef.GPU()
    )

    with Graph("vector_add", input_types=(input_type, input_type)) as graph:
        vector1, vector2 = graph.inputs[0].tensor, graph.inputs[1].tensor
        output = vector1 + vector2
        graph.output(output)

    session = engine.InferenceSession(devices=[Accelerator()])
    model = session.load(graph)
    model._export_mef(output_path)


if __name__ == "__main__":
    main()
