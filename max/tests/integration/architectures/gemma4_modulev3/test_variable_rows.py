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
"""Graph-build test for the variable-logits row gather on an accelerator.

``return_n_logits`` is a host input while the row offsets live on the
device, so a gather that builds its row range on the host fails at trace
time with "Input values must be on the same device". An eager CPU run
cannot see this, so the target and draft share ``gather_variable_rows``
and this traces it with device-resident offsets.
"""

from __future__ import annotations

from max.driver import CPU, Accelerator
from max.dtype import DType
from max.experimental import compilation
from max.experimental.sharding import DeviceMesh, TensorLayout
from max.experimental.tensor import Tensor
from max.pipelines.speculative.driver import gather_variable_rows


def test_gather_variable_rows_traces_with_device_offsets() -> None:
    device = Accelerator(0)
    mesh = DeviceMesh((device,), (1,), ("tp",))

    def gather(
        h: Tensor, input_row_offsets: Tensor, return_n_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        return gather_variable_rows(h, input_row_offsets, return_n_logits, mesh)

    staged = compilation.stage(gather)(
        TensorLayout(DType.bfloat16, ["rows", 8], device),
        TensorLayout(DType.uint32, ["offsets_len"], device),
        TensorLayout(DType.int64, [1], CPU()),
    )
    assert staged is not None
