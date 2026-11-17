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
"""State-space and linear-attention kernels as ``Tensor`` ops.

These have no sharding rule: on distributed inputs each runs once per
device. The ``kda_chunk_*`` predicates say on the host whether this build
and device serve the chunked kernels.
"""

from max import functional as F
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_fwd as _gated_delta_conv1d_fwd,
)
from max.nn._graph_ops.state_space import (
    gated_delta_conv1d_verify_fwd as _gated_delta_conv1d_verify_fwd,
)
from max.nn._graph_ops.state_space import (
    gated_delta_recurrence_fwd as _gated_delta_recurrence_fwd,
)
from max.nn._graph_ops.state_space import (
    gated_delta_recurrence_verify_ring_fwd as _gated_delta_recurrence_verify_ring_fwd,
)
from max.nn._graph_ops.state_space import kda_chunk as _kda_chunk
from max.nn._graph_ops.state_space import (
    kda_chunk_is_registered,
    kda_chunk_rows_servable,
    kda_chunk_supports_head_dims,
    kda_verify_ring_kgroup_conv_servable,
)
from max.nn._graph_ops.state_space import kda_chunk_rows as _kda_chunk_rows
from max.nn._graph_ops.state_space import kda_decode as _kda_decode
from max.nn._graph_ops.state_space import kda_state_fold as _kda_state_fold
from max.nn._graph_ops.state_space import kda_verify_ring as _kda_verify_ring
from max.nn._graph_ops.state_space import (
    kda_verify_ring_kgroup_conv as _kda_verify_ring_kgroup_conv,
)

gated_delta_conv1d_fwd = F.functional(_gated_delta_conv1d_fwd)
gated_delta_conv1d_verify_fwd = F.functional(_gated_delta_conv1d_verify_fwd)
gated_delta_recurrence_fwd = F.functional(_gated_delta_recurrence_fwd)
gated_delta_recurrence_verify_ring_fwd = F.functional(
    _gated_delta_recurrence_verify_ring_fwd
)
kda_chunk = F.functional(_kda_chunk)
kda_chunk_rows = F.functional(_kda_chunk_rows)
kda_decode = F.functional(_kda_decode)
kda_state_fold = F.functional(_kda_state_fold)
kda_verify_ring = F.functional(_kda_verify_ring)
kda_verify_ring_kgroup_conv = F.functional(_kda_verify_ring_kgroup_conv)


__all__ = [
    "gated_delta_conv1d_fwd",
    "gated_delta_conv1d_verify_fwd",
    "gated_delta_recurrence_fwd",
    "gated_delta_recurrence_verify_ring_fwd",
    "kda_chunk",
    "kda_chunk_is_registered",
    "kda_chunk_rows",
    "kda_chunk_rows_servable",
    "kda_chunk_supports_head_dims",
    "kda_decode",
    "kda_state_fold",
    "kda_verify_ring",
    "kda_verify_ring_kgroup_conv",
    "kda_verify_ring_kgroup_conv_servable",
]
