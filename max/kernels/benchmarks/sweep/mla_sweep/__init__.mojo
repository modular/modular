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
"""Shared pieces of the sparse MLA sweeps.

- `cells`: the `batch:seq:cache` cell lists the sweeps take.
- `checks`: output agreement between two arms, reduced on the device, and
  the capture check for graph-timed launches.
- `launch`: forced split-K dispatch scalars, per-token top-k lengths,
  clock-stamp dumps and the PDL predecessor spin.
- `nvfp4`: the packed NVFP4 latent cache, its scale pool and the FP8 mirror
  holding its exact dequantization.
"""

from .cells import cells_from_arg, parse_names
from .checks import Agreement, agreement, check_agreement, verify_capture
from .launch import (
    dump_stamps,
    early_trigger_spin,
    fill_topk_lengths,
    force_split,
)
from .nvfp4 import (
    PACKED_ROW,
    QUANT_GRANULARITY,
    SF_PER_ROW,
    PackedLatent,
    interleave_index_rows,
    permute_blocks,
)
