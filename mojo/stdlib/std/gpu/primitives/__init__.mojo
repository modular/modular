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
"""GPU primitives package - warp, block, cluster, and grid-level operations.

This package provides low-level GPU execution primitives at various levels
of the GPU hierarchy:

- **warp**: Warp-level operations (shuffle, reduce, broadcast)
- **block**: Block-level operations (reductions across thread blocks)
- **cluster**: Cluster-level synchronization (SM90+)
- **grid_controls**: Grid dependency control (Hopper PDL)
- **id**: Thread/block/grid indexing and dimensions

These primitives form the foundation for GPU kernel development.
"""

# Cluster operations (SM90+)
from .cluster import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_arrive_relaxed,
    cluster_sync,
    cluster_sync_relaxed,
    cluster_wait,
    elect_one_sync,
)

# Grid control operations (Hopper PDL)
from .grid_controls import (
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

# Thread/block/grid indexing
from .id import (
    block_dim,
    block_dim_uint,
    block_id_in_cluster,
    block_idx,
    block_idx_uint,
    cluster_dim,
    cluster_idx,
    global_idx,
    global_idx_uint,
    grid_dim,
    grid_dim_uint,
    lane_id,
    lane_id_uint,
    sm_id,
    thread_idx,
    thread_idx_uint,
    warp_id,
    warp_id_uint,
)
