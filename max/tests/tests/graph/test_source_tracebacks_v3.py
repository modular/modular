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

"""Source tracebacks stop at the caller's call into the graph API."""

from collections.abc import Iterator

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType


@pytest.fixture
def source_tracebacks() -> Iterator[None]:
    debug = InferenceSession.debug
    saved = (debug.source_tracebacks, debug.source_tracebacks_hide_internal)
    debug.source_tracebacks = True
    yield
    debug.source_tracebacks, debug.source_tracebacks_hide_internal = saved


class _SquareSum(Module[[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return F.sum(x * x, axis=-1)


@pytest.mark.usefixtures("source_tracebacks")
def test_module_v3_tracebacks_stop_at_the_op_call() -> None:
    staged, _ = _SquareSum().stage(
        TensorType(DType.float32, [4, 8], device=CPU())
    )
    text = staged.graph.module._to_mlir_str(source_locations=True)
    # The `forward` line that created each op is kept; the op wrappers and
    # `max.graph.ops` below it are not.
    assert "in forward" in text
    assert "spmd_ops.py" not in text
    assert "max/graph/ops/" not in text
