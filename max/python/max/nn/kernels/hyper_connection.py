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

"""Hyper-connection kernels as ``Tensor`` ops, with their sharding rules."""

from max import functional as F
from max.nn._graph_ops import hyper_connection_gates as _hyper_connection_gates
from max.nn._graph_ops import mhc_post as _mhc_post
from max.nn._graph_ops import mhc_pre as _mhc_pre
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import R
from max.tensor.sharding.types import TensorLayout

hyper_connection_gates = F.functional(_hyper_connection_gates)


def _mhc_post_rule(
    streams: TensorLayout,
    y: TensorLayout,
    post: TensorLayout,
    comb: TensorLayout,
) -> list[AxisAssignment]:
    # The mHC streams are replicated on every device, as a single residual is,
    # so the write-back only runs on replicated inputs.
    return [AxisAssignment((R, R, R, R), (R,))]


mhc_post = F.functional(_mhc_post, rule=_mhc_post_rule)


mhc_pre = F.functional(_mhc_pre)


__all__ = [
    "hyper_connection_gates",
    "mhc_post",
    "mhc_pre",
]
