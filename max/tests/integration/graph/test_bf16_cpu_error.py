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

import platform

import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


@pytest.mark.skipif(
    platform.machine() not in ["arm64", "aarch64"],
    reason="This test validates BF16 support on ARM CPU architecture",
)
def test_bf16_cpu_input(session: InferenceSession) -> None:
    input_type = TensorType(
        dtype=DType.bfloat16, shape=["dim"], device=DeviceRef.CPU()
    )
    output_type = DType.float32
    with Graph("cast", input_types=[input_type]) as graph:
        graph.output(graph.inputs[0].tensor.cast(output_type))

    model = session.load(graph)
    assert model is not None
