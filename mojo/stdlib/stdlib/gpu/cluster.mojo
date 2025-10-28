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
"""GPU cluster operations (deprecated - use `gpu.primitives.cluster` or `gpu`).

This module is deprecated. For new code, import cluster operations from the
`gpu` package or `gpu.primitives.cluster` module:

```mojo
# Deprecated:
from gpu.cluster import cluster_sync, cluster_arrive

# Recommended (import from top-level gpu package):
from gpu import cluster_sync, cluster_arrive

# Or import the module:
from gpu.primitives import cluster
```

This module provides cluster-level synchronization operations for NVIDIA
SM90+ GPUs (Hopper architecture and newer).
"""

# Re-export all symbols from primitives.cluster for backward compatibility
from .primitives.cluster import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_arrive_relaxed,
    cluster_mask_base,
    cluster_sync,
    cluster_sync_acquire,
    cluster_sync_relaxed,
    cluster_sync_release,
    cluster_wait,
    clusterlaunchcontrol_query_cancel_get_first_ctaid,
    clusterlaunchcontrol_query_cancel_get_first_ctaid_v4,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_try_cancel,
    elect_one_sync,
    elect_one_sync_with_mask,
)
