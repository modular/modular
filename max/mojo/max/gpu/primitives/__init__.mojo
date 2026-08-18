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
# TODO(MSTDL-2788): Restore the `warp` and `id` entries below once those modules
# move here from `std/gpu/primitives/`. They are listed in
# `std/gpu/primitives/__init__.mojo` and belong in this list, but naming them
# here today advertises modules this package does not contain.
"""GPU primitives package - block, cluster, and grid-level operations.

This package provides low-level GPU execution primitives at various levels
of the GPU hierarchy:

- **block**: Block-level operations (reductions across thread blocks)
- **cluster**: Cluster-level synchronization (SM90+)
- **grid_controls**: Grid dependency control (Hopper PDL)

These primitives form the foundation for GPU kernel development.
"""

# Grid control operations (Hopper PDL)
from .grid_controls import (
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

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
