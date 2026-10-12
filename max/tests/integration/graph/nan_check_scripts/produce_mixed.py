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
"""Subprocess script: build a div graph and execute with inputs that produce
both NaN and Inf.

Inputs: [0.0, 1.0, 2.0, 3.0] / [0.0, 0.0, 1.0, 1.0]
Expected output: [NaN, Inf, 2.0, 3.0]

With max-debug.nan-check enabled, this should abort with a diagnostic showing
1 NaN, 1 Inf of 4 total elements.

Set NAN_CHECK_TEST_DEVICE=gpu to run on GPU instead of CPU.
"""

import os

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

use_gpu = os.environ.get("NAN_CHECK_TEST_DEVICE", "cpu") == "gpu"
if use_gpu:
    assert accelerator_count() > 0, "GPU requested but no accelerator available"
    device = Accelerator()
    device_ref = DeviceRef.GPU()
else:
    device = CPU()
    device_ref = DeviceRef.CPU()

input_type = TensorType(dtype=DType.float32, shape=[4], device=device_ref)
with Graph("mixed_graph", input_types=[input_type, input_type]) as g:
    result = ops.div(g.inputs[0], g.inputs[1])
    g.output(result)

session = InferenceSession(devices=[device])
model = session.load(g)

# [0/0, 1/0, 2/1, 3/1] = [NaN, Inf, 2.0, 3.0]
numerator = Buffer.from_numpy(
    np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
).to(model.input_devices[0])
denominator = Buffer.from_numpy(
    np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
).to(model.input_devices[0])

model.execute(numerator, denominator)
