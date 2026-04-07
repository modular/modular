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
"""A simple add+relu model used as a subprocess by test_uninit_check_e2e.py.

Builds a Graph, compiles it via InferenceSession, runs inference, and
verifies the output.  If MODULAR_MAX_UNINITIALIZED_READ_CHECK is set in
the environment, InferenceSession automatically enables the debug
allocator poison and the Mojo MOJO_STDLIB_SIMD_UNINIT_CHECK define.
"""

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# Build a simple add+relu graph.
with Graph(
    "add_relu",
    input_types=[
        TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
        TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
    ],
) as graph:
    a, b = (inp.tensor for inp in graph.inputs)
    result = ops.relu(ops.add(a, b))
    graph.output(result)

session = InferenceSession(devices=[CPU()])
model = session.load(graph)

a = np.ones((4, 4), dtype=np.float32) * 2.0
b = np.ones((4, 4), dtype=np.float32) * 3.0
outputs = model.execute(a, b)

out = outputs[0].to_numpy()
expected = np.maximum(a + b, 0)
assert np.allclose(out, expected), f"Output mismatch: {out} != {expected}"

print("ALL CHECKS PASSED")
