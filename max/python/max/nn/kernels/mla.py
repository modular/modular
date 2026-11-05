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

from max import functional as F
from max.nn._graph_ops import flare_mla_prefill_plan as _flare_mla_prefill_plan
from max.nn._graph_ops import mla_decode_graph as _mla_decode_graph
from max.nn._graph_ops import mla_fp8_index_top_k as _mla_fp8_index_top_k
from max.nn._graph_ops import (
    mla_fp8_index_top_k_physical as _mla_fp8_index_top_k_physical,
)
from max.nn._graph_ops import (
    mla_prefill_decode_graph as _mla_prefill_decode_graph,
)
from max.nn._graph_ops import mla_prefill_graph as _mla_prefill_graph
from max.nn._graph_ops import (
    store_mla_latent_ragged as _store_mla_latent_ragged,
)

flare_mla_prefill_plan = F.functional(_flare_mla_prefill_plan)
mla_prefill_graph = F.functional(_mla_prefill_graph)
mla_decode_graph = F.functional(_mla_decode_graph)
mla_prefill_decode_graph = F.functional(_mla_prefill_decode_graph)
mla_fp8_index_top_k = F.functional(_mla_fp8_index_top_k)
mla_fp8_index_top_k_physical = F.functional(_mla_fp8_index_top_k_physical)


store_mla_latent_ragged = F.functional(_store_mla_latent_ragged)


__all__ = [
    "flare_mla_prefill_plan",
    "mla_decode_graph",
    "mla_fp8_index_top_k",
    "mla_fp8_index_top_k_physical",
    "mla_prefill_decode_graph",
    "mla_prefill_graph",
    "store_mla_latent_ragged",
]
