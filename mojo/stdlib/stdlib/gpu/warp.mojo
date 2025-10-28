# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""GPU warp-level operations (deprecated - use `gpu.primitives.warp` module).

This module is deprecated. For new code, import the warp module from
`gpu.primitives`:

```mojo
# Deprecated:
from gpu.warp import shuffle, reduce, broadcast

# Recommended (import the module):
from gpu.primitives import warp
# Then use: warp.shuffle(), warp.reduce(), warp.broadcast()

# Or import specific functions:
from gpu.primitives.warp import shuffle, reduce, broadcast
```

This module provides warp-level primitives including shuffle operations,
reductions, broadcasts, and synchronization.
"""

# Re-export all public symbols from primitives.warp for backward compatibility
from .primitives.warp import (
    ReductionMethod,
    WARP_SIZE,
    broadcast,
    lane_group_max,
    lane_group_max_and_broadcast,
    lane_group_min,
    lane_group_reduce,
    lane_group_sum,
    lane_group_sum_and_broadcast,
    max,
    min,
    prefix_sum,
    reduce,
    shuffle_down,
    shuffle_idx,
    shuffle_up,
    shuffle_xor,
    sum,
    vote,
)
