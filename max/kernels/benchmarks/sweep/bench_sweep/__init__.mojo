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
"""Building blocks for benchmark sweeps that run many configs in one process.

A sweep pays process start, device init, compilation and fixture setup once,
then reuses them across every config:

- `driver`: runtime axes, config filtering, and planning fixtures at the
  maximum over a sweep, split by a device memory budget.
- `backend`: the target-specific queries a sweep makes -- fill, timing,
  device limits, throttle -- with the answers for the targets the library
  knows; a plugin-backed target supplies its own from its plugin directory.
- `fixtures`: a paged KV cache whose page table and blocks every config
  shares.
- `generators`: deterministic device fills with bit-exact CPU references.
- `timing`: `PhaseTimer`, which attributes wall time to each phase so the
  saving is measured rather than assumed.
- `harness`: `SweepHarness`, the measurement policy (target time, a cold
  cache by rotating input copies, throttle check) and the JSON output.

See `docs/plans/2026-10-12-benchmark-sweep-library.md`.
"""

from .backend import (
    SweepBackend,
    auto_device_events,
    auto_evict_bytes,
    device_fill_supported,
    throttle_state,
)
from .driver import (
    Config,
    Footprint,
    Sweep,
    SweepGroup,
    arm_enabled,
    device_budget,
    dtype_arm_enabled,
    parse_int_list,
    plan_groups,
)
from .fixtures import PagedBlocks, PageTable
from .generators import (
    PageLayout,
    PagePermutation,
    affine_at,
    affine_host,
    coprime_near,
    distinct_stride,
    fill_affine,
    fill_linear_grade,
    fill_lut_view,
    fill_page_table,
    fill_sparse_indices,
    fill_uniform,
    fill_uniform_host_rand,
    linear_grade_at,
    linear_grade_host,
    lut_view_at,
    lut_view_host,
    mix32,
    page_table_host,
    sparse_indices_host,
    sparse_slot,
    sparse_stride,
    uniform_at,
    uniform_bits,
    uniform_host,
)
from .harness import (
    Fields,
    MeasurePolicy,
    Rotation,
    SweepHarness,
)
from .timing import (
    PhaseTimer,
    bench_charged,
    charge_last_bench,
    seconds_since_process_start,
)
