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

"""Generate MEF file for testing debug printing in C API."""

import sys

from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_mef_path>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]

    # Create graph with debug print operations
    input_type = TensorType(
        dtype=DType.float32, shape=(8,), device=DeviceRef.CPU()
    )

    with Graph("debug_print", input_types=(input_type, input_type)) as graph:
        v1, v2 = graph.inputs[0].tensor, graph.inputs[1].tensor

        # Print inputs
        v1.print("input_vector_1")
        v2.print("input_vector_2")

        # Compute and print intermediate
        intermediate = v1 + v2
        intermediate.print("sum_result")

        # Final computation
        output = intermediate * 2.0
        output.print("final_output")

        graph.output(output)

    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
    model._export_mef(output_path)


if __name__ == "__main__":
    main()
