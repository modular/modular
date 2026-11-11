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
"""Scatter kernels as ``Tensor`` ops.

These have no sharding rule: on distributed inputs each runs once per
device.
"""

from max import functional as F
from max.nn._graph_ops import (
    scatter_nd_skip_oob_indices as _scatter_nd_skip_oob_indices,
)

scatter_nd_skip_oob_indices = F.functional(_scatter_nd_skip_oob_indices)


__all__ = [
    "scatter_nd_skip_oob_indices",
]
