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
"""Where a rebind claim lands depends on the tensor's rank, not its shape.

Framework semantics, not a layer test: `rebind_mapping` trusts the mapping
it is handed -- it checks the mesh and nothing else -- so a placement list
written for one rank still applies to a tensor of another rank, and the
named mesh axis lands on the tensor axis at the same position in the list.
These cases record where TP resolves for the two shapes the KDA verify
wrapper can hand its layer, the raw 4-D `[1, T, heads_per_device,
head_dim]` readout and a 3-D `[T, heads_per_device, head_dim]` view of it,
so a claim can be matched to the rank of the tensor it is applied to. On
the 3-D view a claim written for the 4-D rank puts TP on `head_dim`, and
the reshape to the global head count then needs a reshard the region
refuses; matched to the 3-D rank, the same reshape is a global no-op.

The layer's own claim is gated elsewhere: `spec_kda_rollback_tp2_tests_2gpu`
rebuilds it on a resolving 2-device mesh, so a rank slip there raises.
"""

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.functional import transfer_to
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Replicated,
    ShardingError,
    Unknown,
)

TP = "tp"
HEADS_PER_DEVICE, HEAD_DIM = 12, 128
GLOBAL_HEADS = HEADS_PER_DEVICE * 8


def _tp8() -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(8)), (8,), (TP,))


def _unknown_local(mesh: DeviceMesh, shape: tuple[int, ...]) -> Tensor:
    """The rule-less dispatch's joined result: local shards, no placement.

    Each device holds its own op output; the buffers' content is irrelevant
    to the placement machinery, so replicated buffers stand in.
    """
    replicated = transfer_to(
        Tensor(np.zeros(shape, dtype=np.float32)),
        DeviceMapping(mesh, (Replicated(),)),
    )
    return replicated.rebind_mapping(DeviceMapping(mesh, (Unknown(),)))


def test_four_entry_claim_on_readout_3d_shards_head_dim() -> None:
    """A four-entry claim on the 3-D readout puts TP on `head_dim`, so the
    reshape to the global head count needs a reshard."""
    mesh = _tp8()
    total_seq_len = 6
    normed = _unknown_local(mesh, (total_seq_len, HEADS_PER_DEVICE, HEAD_DIM))
    claimed = normed.rebind_mapping(NamedMapping(mesh, (None, None, TP, None)))
    with pytest.raises(ShardingError, match="needs a resharding"):
        F.reshape(
            claimed,
            [total_seq_len, GLOBAL_HEADS, HEAD_DIM],
        )


def test_four_entry_claim_on_readout_4d_resolves() -> None:
    """The same claim on the 4-D readout puts TP on the head axis, and the
    reshape resolves."""
    mesh = _tp8()
    total_seq_len = 6
    o = _unknown_local(mesh, (1, total_seq_len, HEADS_PER_DEVICE, HEAD_DIM))
    o = o.rebind_mapping(NamedMapping(mesh, (None, None, TP, None)))
    merged = F.reshape(o, [total_seq_len, GLOBAL_HEADS, HEAD_DIM])
    assert merged.shape == [total_seq_len, GLOBAL_HEADS, HEAD_DIM]


def test_three_entry_claim_on_readout_3d_resolves() -> None:
    """A claim matched to the 3-D rank puts TP on the head axis, and the
    reshape is a global no-op."""
    mesh = _tp8()
    total_seq_len = 6
    normed = _unknown_local(mesh, (total_seq_len, HEADS_PER_DEVICE, HEAD_DIM))
    claimed = normed.rebind_mapping(NamedMapping(mesh, (None, TP, None)))
    merged = F.reshape(claimed, [total_seq_len, GLOBAL_HEADS, HEAD_DIM])
    assert merged.shape == [total_seq_len, GLOBAL_HEADS, HEAD_DIM]
