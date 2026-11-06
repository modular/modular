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

"""Multi-latent attention kernels as ``Tensor`` ops.

These have no sharding rule: on distributed inputs each runs once per
device.
"""

from typing import Any

from max import functional as F
from max.nn._graph_ops import cp_merge_topk as _cp_merge_topk
from max.nn._graph_ops import flare_mla_prefill_plan as _flare_mla_prefill_plan
from max.nn._graph_ops import mla_decode_graph as _mla_decode_graph
from max.nn._graph_ops import mla_decode_graph_cp as _mla_decode_graph_cp
from max.nn._graph_ops import mla_fp8_index_top_k as _mla_fp8_index_top_k
from max.nn._graph_ops import (
    mla_fp8_index_top_k_physical as _mla_fp8_index_top_k_physical,
)
from max.nn._graph_ops import (
    mla_fp8_index_top_k_physical_cp as _mla_fp8_index_top_k_physical_cp,
)
from max.nn._graph_ops import (
    mla_prefill_decode_graph as _mla_prefill_decode_graph,
)
from max.nn._graph_ops import mla_prefill_graph as _mla_prefill_graph
from max.nn._graph_ops import (
    store_mla_latent_ragged as _store_mla_latent_ragged,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Unknown

flare_mla_prefill_plan = F.functional(_flare_mla_prefill_plan)
mla_prefill_graph = F.functional(_mla_prefill_graph)
mla_decode_graph = F.functional(_mla_decode_graph)
mla_prefill_decode_graph = F.functional(_mla_prefill_decode_graph)
mla_fp8_index_top_k = F.functional(_mla_fp8_index_top_k)
mla_fp8_index_top_k_physical = F.functional(_mla_fp8_index_top_k_physical)
mla_fp8_index_top_k_physical_cp = F.functional(_mla_fp8_index_top_k_physical_cp)
cp_merge_topk = F.functional(_cp_merge_topk)
_mla_decode_graph_cp_per_device = F.functional(_mla_decode_graph_cp)


def mla_decode_graph_cp(q: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
    """Runs :func:`~max.nn._graph_ops.mla_decode_graph_cp` on each device.

    A decode context-parallel rank's partial output and LSE cover ``q``'s
    tokens but differ across the ranks of its group, so both are ``Unknown``
    on the DCP mesh axis and keep ``q``'s placements on the other axes,
    where a data-parallel replica's tokens are its own.
    """
    # context_parallel imports this module.
    from max.nn.common_layers.context_parallel import dcp_axis

    out, lse = _mla_decode_graph_cp_per_device(q=q, **kwargs)
    axis = dcp_axis(q.mesh)
    partial = DeviceMapping(
        q.mesh,
        tuple(
            Unknown() if i == axis else p for i, p in enumerate(q.placements)
        ),
    )
    return out.rebind_mapping(partial), lse.rebind_mapping(partial)


store_mla_latent_ragged = F.functional(_store_mla_latent_ragged)


__all__ = [
    "cp_merge_topk",
    "flare_mla_prefill_plan",
    "mla_decode_graph",
    "mla_decode_graph_cp",
    "mla_fp8_index_top_k",
    "mla_fp8_index_top_k_physical",
    "mla_fp8_index_top_k_physical_cp",
    "mla_prefill_decode_graph",
    "mla_prefill_graph",
    "store_mla_latent_ragged",
]
