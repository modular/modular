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
"""GPU thread and block indexing (deprecated - use `gpu` package directly).

This module is deprecated. For new code, import these symbols directly from
the `gpu` package instead:

```mojo
# Deprecated:
from gpu.id import block_idx, thread_idx

# Recommended:
from gpu import block_idx, thread_idx
```

This module provides GPU thread and block indexing functionality, including
aliases and functions for accessing GPU grid, block, thread and cluster
dimensions and indices.
"""

# Re-export all symbols from primitives.id for backward compatibility
from .primitives.id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    cluster_dim,
    cluster_idx,
    global_idx,
    grid_dim,
    lane_id,
    sm_id,
    thread_idx,
    warp_id,
)
