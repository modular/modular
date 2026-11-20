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
"""Vision encoder kernels as ``Tensor`` ops.

These have no sharding rule: on distributed inputs each runs once per
device.
"""

from max import functional as F
from max.nn._graph_ops import spatial_merge as _spatial_merge

spatial_merge = F.functional(_spatial_merge)


__all__ = [
    "spatial_merge",
]
