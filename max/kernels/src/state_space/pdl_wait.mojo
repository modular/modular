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
"""Grid-dependency wait that keeps the loads after it behind it (SM90+)."""

from max.gpu.primitives.grid_controls import (
    _SUPPORT_PDL_LAUNCH,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier


@always_inline
def wait_on_producers():
    """Waits on the predecessor grids, then synchronizes the CTA.

    The CTA barrier is a defence, not a measured requirement. An earlier layout
    of the Mamba-2 kernels showed the assembler issuing `LDG.E.CONSTANT` loads
    (what a tensor with an immutable origin loads with) above
    `griddepcontrol.wait` once their address was known, so they read what the
    predecessor had not written yet. A CTA barrier is an ordering point those
    loads are not moved across. In the current kernels (the conv decode launch,
    the decode and staged scan tiles, both norms) the SASS has no global load
    above the wait with or without the barrier, and no test exercises the
    barrier. That holds for the tile in use: re-check the SASS when a kernel's
    tile or load order changes. Call this where every thread of the CTA arrives,
    before any early return.
    """
    wait_on_dependent_grids()
    comptime if _SUPPORT_PDL_LAUNCH:
        barrier()
