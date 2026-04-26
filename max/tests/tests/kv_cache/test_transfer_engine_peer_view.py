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

"""Unit tests for KVTransferEngine's peer-view shape resolver.

``resolve_peer_view`` decides whether a local or remote ``[dp][tp]`` grid
is reinterpreted as ``[dp*tp][1]`` so a prefill worker at (DP=m, TP=n)
can connect to a decode worker at (DP=m*n, TP=1).
"""

from __future__ import annotations

import pytest
from max.kv_cache.paged_kv_cache.transfer_engine import resolve_peer_view


@pytest.mark.parametrize(
    "name,local_dp,local_tp,local_rep,remote_dp,remote_tp,remote_rep,"
    "flatten_local,flatten_remote,effective_dp",
    [
        ("homogeneous_match", 2, 4, True, 2, 4, True, False, False, 2),
        ("local_flattens_mla_to_dp", 1, 8, True, 8, 1, True, True, False, 8),
        ("remote_flattens_dp_to_mla", 8, 1, True, 1, 8, True, False, True, 8),
        ("homogeneous_non_mla", 4, 2, False, 4, 2, False, False, False, 4),
    ],
)
def test_resolve_peer_view_accepted(
    name: str,
    local_dp: int,
    local_tp: int,
    local_rep: bool,
    remote_dp: int,
    remote_tp: int,
    remote_rep: bool,
    flatten_local: bool,
    flatten_remote: bool,
    effective_dp: int,
) -> None:
    view = resolve_peer_view(
        local_dp=local_dp,
        local_tp=local_tp,
        local_replicate=local_rep,
        remote_dp=remote_dp,
        remote_tp=remote_tp,
        remote_replicate=remote_rep,
    )
    assert view.flatten_local is flatten_local
    assert view.flatten_remote is flatten_remote
    assert view.effective_dp == effective_dp


@pytest.mark.parametrize(
    "name,local_dp,local_tp,local_rep,remote_dp,remote_tp,remote_rep",
    [
        # Heterogeneous shapes without MLA replication on either side.
        ("neither_replicates", 1, 8, False, 8, 1, False),
        # Local dp*tp=4 but remote dp=8.
        ("dp_product_mismatch", 1, 4, True, 8, 1, True),
        # Both sides TP>1 with mismatched shapes.
        ("both_tp_gt_1_mismatch", 2, 4, True, 4, 2, True),
        # TP=1 sides claiming to replicate (nonsensical; caller bug).
        ("tp1_replicate_nonsense", 8, 1, True, 1, 1, True),
    ],
)
def test_resolve_peer_view_rejected(
    name: str,
    local_dp: int,
    local_tp: int,
    local_rep: bool,
    remote_dp: int,
    remote_tp: int,
    remote_rep: bool,
) -> None:
    with pytest.raises(ValueError, match="Incompatible"):
        resolve_peer_view(
            local_dp=local_dp,
            local_tp=local_tp,
            local_replicate=local_rep,
            remote_dp=remote_dp,
            remote_tp=remote_tp,
            remote_replicate=remote_rep,
        )
