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
from max.driver import CPU
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor, default_device


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
