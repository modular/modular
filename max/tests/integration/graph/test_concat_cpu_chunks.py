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

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_concat_reads_each_vector_from_one_input_on_cpu(
    session: InferenceSession,
) -> None:
    """A fused CPU concat never reads a vector across two inputs.

    The concat limits the width to 2, the largest power of two dividing both
    extents, which keeps each vector inside one input only if vectors start
    on multiples of 2. The CPU 1-D `elementwise` used to split these 65,540
    elements into 3 worker chunks of 21,847, so the second worker's vectors
    started at odd offsets and the one at 32,769 read element 32,770 from the
    first input instead of the second.
    """
    n = 32770
    device = DeviceRef.CPU()
    with Graph(
        "concat_cpu_chunks",
        input_types=[
            TensorType(DType.float32, [n], device=device),
            TensorType(DType.float32, [n], device=device),
        ],
    ) as graph:
        a, b = graph.inputs
        graph.output(ops.concat([a.tensor, b.tensor + b.tensor], axis=0))

    a_data = np.ones(n, dtype=np.float32)
    b_data = np.ones(n, dtype=np.float32)
    (actual,) = run_and_verify_fusion(
        session,
        graph,
        a_data,
        b_data,
        fused=r"^(?=.*mo\.concat)(?=.*mo\.(add|mul))",
    )

    expected = np.concatenate([a_data, 2 * b_data])
    np.testing.assert_equal(actual, expected)
