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

"""Pure-metadata tests for normalization placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping
from max.experimental.sharding.rules.norm import normalization_rule
from max.experimental.sharding.types import TensorLayout

from rules._fixtures import MESH_1D, M, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Normalization rule
# ═════════════════════════════════════════════════════════════════════════


class TestNormalizationRule:
    def test_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = normalization_rule(x, gamma, beta)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        """S(0) on [B, H] with weight shape [H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = normalization_rule(x, gamma, beta)
        assert out.to_placements() == (S(0),)

    def test_norm_dim_sharded_raises(self) -> None:
        """Sharding on the hidden dim (being normalized) must error."""
        x = _layout(M(MESH_1D, S(1)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        with pytest.raises(ValueError, match="cannot normalize"):
            normalization_rule(x, gamma, beta)

    def test_3d_input_2d_weight(self) -> None:
        """[B, S, H] with weight [S, H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = normalization_rule(x, gamma, beta)
        assert out.to_placements() == (S(0),)

    def test_3d_input_sharded_seq_raises(self) -> None:
        """[B, S, H] with S(1) (seq dim) when weight is [S, H]."""
        x = _layout(M(MESH_1D, S(1)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        with pytest.raises(ValueError, match="cannot normalize"):
            normalization_rule(x, gamma, beta)
