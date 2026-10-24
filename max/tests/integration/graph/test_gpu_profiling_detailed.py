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
"""Runs a GPU graph with detailed kernel profiling and no profiler attached.

The GPU CI executors, like serving images built on the CUDA base image, ship no
NVTX library. With detailed profiling every kernel launch opens an NVTX range,
and the first one used to abort the process with "symbol not found:
nvtxRangeStartEx".
"""

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_gpu_graph_runs_with_detailed_profiling() -> None:
    session = InferenceSession(devices=[Accelerator()])
    session.gpu_profiling("detailed")

    input_type = TensorType(DType.float32, [4, 4], device=DeviceRef.GPU())
    with Graph(
        "detailed_profiling_matmul", input_types=[input_type, input_type]
    ) as graph:
        lhs, rhs = graph.inputs
        graph.output(ops.matmul(lhs.tensor, rhs.tensor))
    model = session.load(graph)

    lhs_np = np.arange(16, dtype=np.float32).reshape(4, 4)
    rhs_np = np.arange(16, 32, dtype=np.float32).reshape(4, 4)
    device = model.input_devices[0]
    result = model(
        Buffer.from_numpy(lhs_np).to(device),
        Buffer.from_numpy(rhs_np).to(device),
    )[0]
    np.testing.assert_allclose(result.to_numpy(), lhs_np @ rhs_np)
