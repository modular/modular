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
"""Builds and runs a GPU-assigned graph whose execute method reads
uninitialized device memory (on the CPU side).

Used as a subprocess by test_uninit_check_e2e.py.  When
MODULAR_MAX_UNINITIALIZED_READ_CHECK=true is set, InferenceSession
enables the debug allocator (which poisons device memory with canonical
qNaN) and the MOJO_STDLIB_SIMD_UNINIT_CHECK define.  The custom
kernel's execute method reads from the output tensor before writing,
hitting the allocator-poisoned memory and triggering the abort.

Expects the UNINIT_OPS_PATH environment variable to point to the
compiled read_uninit_op Mojo package.
"""

import os
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

ops_path = Path(os.environ["UNINIT_OPS_PATH"])

tensor_type = TensorType(DType.float32, [1], device=DeviceRef.GPU())

with Graph(
    "trigger_uninit",
    input_types=[tensor_type],
    custom_extensions=[ops_path],
) as graph:
    result = ops.custom(
        "read_uninit_output",
        device=DeviceRef.GPU(),
        values=[graph.inputs[0]],
        out_types=[tensor_type],
    )[0]
    graph.output(result)

gpu = Accelerator()
session = InferenceSession(devices=[gpu, CPU()])
model = session.load(graph)

dummy_input = Buffer.from_numpy(np.zeros((1,), dtype=np.float32)).to(gpu)
model.execute(dummy_input)

# Should not reach here when the check is enabled.
print("NO ABORT")
