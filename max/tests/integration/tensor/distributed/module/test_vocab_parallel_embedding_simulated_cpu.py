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
"""Tests a vocabulary-split embedding against its whole table, on a simulated CPU mesh."""

from __future__ import annotations

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh


@pytest.mark.parametrize(
    ("vocab_size", "num_devices"),
    # 10 rows on 4 devices split 3, 3, 2, 2.
    [(8, 2), (10, 4)],
)
def test_lookup_matches_the_whole_table(
    vocab_size: int, num_devices: int
) -> None:
    mesh = DeviceMesh(
        devices=tuple(CPU() for _ in range(num_devices)),
        mesh_shape=(num_devices,),
        axis_names=("tp",),
    )
    with default_device(mesh):
        embedding = VocabParallelEmbedding(vocab_size, dim=3)
    assert embedding.weight.is_distributed
    table = embedding.weight.to_numpy()
    ids = np.arange(vocab_size - 1, -1, -1, dtype=np.int64)

    out = embedding(Tensor(ids))

    np.testing.assert_allclose(out.to_numpy(), table[ids], rtol=1e-6)


def test_split_over_the_tp_axis_of_a_dp_tp_mesh() -> None:
    """On a ``(dp, tp)`` mesh the table splits over ``tp`` only.

    Each replica holds the whole table across its four devices, so every
    device of every replica gets the full lookup. Traces the lookup without
    compiling or running it; the lookup values are
    ``test_lookup_matches_the_whole_table``'s.
    """
    mesh = DeviceMesh(
        devices=tuple(CPU() for _ in range(8)),
        mesh_shape=(2, 4),
        axis_names=("dp", "tp"),
    )
    vocab_size, dim = 10, 3
    with F.lazy():
        with default_device(mesh):
            embedding = VocabParallelEmbedding(vocab_size, dim=dim)
        # 10 rows over the 4 tp ranks split 3, 3, 2, 2, the same in each
        # replica.
        assert [int(w.shape[0]) for w in embedding.weight.local_shards] == [
            3,
            3,
            2,
            2,
        ] * 2
        ids = np.arange(vocab_size - 1, -1, -1, dtype=np.int64)
        out = embedding(Tensor(ids))
    assert not out.real
    for shard in out.local_shards:
        assert shard.shape == [vocab_size, dim]
