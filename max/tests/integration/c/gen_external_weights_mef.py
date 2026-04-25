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

"""Generate a MEF with external weights for C API weights registry testing.

The graph computes: output = input + weight
where 'weight' is an external constant provided via the weights registry.
"""

import sys

import numpy as np
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_mef_path>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]

    input_type = TensorType(
        dtype=DType.float32, shape=(4,), device=DeviceRef.CPU()
    )
    weight_type = TensorType(
        dtype=DType.float32, shape=(4,), device=DeviceRef.CPU()
    )

    with Graph("external_weights", input_types=(input_type,)) as graph:
        inp = graph.inputs[0].tensor
        weight = ops.constant_external("my_weight", weight_type)
        output = inp + weight
        graph.output(output)

    # Provide dummy weights for compilation. The actual weight values
    # are supplied at runtime via M_newWeightsRegistry in the C API.
    dummy_weights = {"my_weight": np.zeros(4, dtype=np.float32)}

    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=dummy_weights)
    model._export_mef(output_path)


if __name__ == "__main__":
    main()
