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

"""Pure-metadata tests for matmul and layer_norm placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping, Partial
from max.experimental.sharding.rules.matmul import layer_norm_rule, matmul_rule
from max.experimental.sharding.types import TensorLayout

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Matmul rule -- [M, K] x [K, N] -> [M, N]
# ═════════════════════════════════════════════════════════════════════════


class TestMatmulRule:
    """Tests for matmul_rule on 1D and 2D meshes.

    For [M=4, K=8] x [K=8, N=6]:
      lhs: M=axis0, K=axis1
      rhs: K=axis0, N=axis1
    """

    # -- Trivial ----------------------------------------------------------

    def test_both_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (R,)

    # -- Data parallel: S(M) x R -> S(M) ---------------------------------

    def test_data_parallel(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0),)

    # -- Column TP: R x S(N) -> S(N) -------------------------------------

    def test_column_tp(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(1),)

    # -- Row TP: S(K_lhs) x S(K_rhs) -> Partial --------------------------

    def test_row_tp(self) -> None:
        """S(K=1) on lhs x S(K=0) on rhs -> Partial."""
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (P,)

    # -- Batch parallel ---------------------------------------------------

    def test_batch_parallel_3d(self) -> None:
        """[B, M, K] x [B, K, N] with S(batch=0) on both."""
        lhs = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (2, 8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_batch_sharded_lhs_only(self) -> None:
        """[B, M, K] x [B, K, N] with S(batch=0) on lhs, R on rhs."""
        lhs = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        rhs = _layout(M(MESH_1D, R), (2, 8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0),)

    # -- Bilinear: P x R -> P, R x P -> P --------------------------------

    def test_partial_lhs_replicated_rhs(self) -> None:
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (P,)

    def test_replicated_lhs_partial_rhs(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (P,)

    # -- Error cases ------------------------------------------------------

    def test_partial_partial_raises(self) -> None:
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        with pytest.raises(ValueError, match="Partial"):
            matmul_rule(lhs, rhs)

    def test_partial_sharded_resolves(self) -> None:
        """P x S(K=0) -> resolves P to R, then R x S(K=0) is unsupported."""
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        # Rule resolves Partial to Replicated, then dispatches R x S(K=0).
        args, (_out,) = matmul_rule(lhs, rhs)
        assert isinstance(args[0], DeviceMapping)
        assert args[0].to_placements() == (R,)

    def test_sharded_partial_resolves(self) -> None:
        """S(M=0) x P -> resolves P to R, then S(M=0) x R = S(M=0)."""
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_s_k_lhs_replicated_rhs_raises(self) -> None:
        """S(K) x R is unsupported -- contraction dim split on one side only."""
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        with pytest.raises(NotImplementedError):
            matmul_rule(lhs, rhs)

    def test_replicated_lhs_s_k_rhs_raises(self) -> None:
        """R x S(K) is unsupported -- contraction dim split on one side only."""
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        with pytest.raises(NotImplementedError):
            matmul_rule(lhs, rhs)

    # -- 2D mesh: combined DP + TP ----------------------------------------

    def test_2d_mesh_dp_plus_column_tp(self) -> None:
        """dp=S(M=0), tp=S(N=1): data parallel x column TP."""
        lhs = _layout(M(MESH_2D, S(0), R), (4, 8))
        rhs = _layout(M(MESH_2D, R, S(1)), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0), S(1))

    def test_2d_mesh_dp_plus_row_tp(self) -> None:
        """dp=S(batch=0), tp=S(K_lhs=2) x S(K_rhs=1): batch parallel + row TP -> Partial.

        For [B=2, M=4, K=8] x [B=2, K=8, N=6]:
          K_lhs=axis2, K_rhs=axis1
        """
        lhs = _layout(M(MESH_2D, S(0), S(2)), (2, 4, 8))
        rhs = _layout(M(MESH_2D, S(0), S(1)), (2, 8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(0), Partial())

    # -- Vector matmul ----------------------------------------------------

    def test_vec_mat(self) -> None:
        """[K] x [K, N] with R x S(N=1) -> S(1)."""
        lhs = _layout(M(MESH_1D, R), (8,))
        rhs = _layout(M(MESH_1D, S(1)), (8, 6))
        _, (out,) = matmul_rule(lhs, rhs)
        assert out.to_placements() == (S(1),)


# ═════════════════════════════════════════════════════════════════════════
#  Layer norm rule
# ═════════════════════════════════════════════════════════════════════════


class TestLayerNormRule:
    def test_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = layer_norm_rule(x, gamma, beta, 1e-5)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        """S(0) on [B, H] with weight shape [H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        _, (out,) = layer_norm_rule(x, gamma, beta, 1e-5)
        assert out.to_placements() == (S(0),)

    def test_norm_dim_sharded_raises(self) -> None:
        """Sharding on the hidden dim (being normalized) must error."""
        x = _layout(M(MESH_1D, S(1)), (4, 8))
        gamma = _layout(M(MESH_1D, R), (8,))
        beta = _layout(M(MESH_1D, R), (8,))
        with pytest.raises(ValueError, match="cannot normalize"):
            layer_norm_rule(x, gamma, beta, 1e-5)

    def test_3d_input_2d_weight(self) -> None:
        """[B, S, H] with weight [S, H]: norm dims start at 1."""
        x = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = layer_norm_rule(x, gamma, beta, 1e-5)
        assert out.to_placements() == (S(0),)

    def test_3d_input_sharded_seq_raises(self) -> None:
        """[B, S, H] with S(1) (seq dim) when weight is [S, H]."""
        x = _layout(M(MESH_1D, S(1)), (2, 4, 8))
        gamma = _layout(M(MESH_1D, R), (4, 8))
        beta = _layout(M(MESH_1D, R), (4, 8))
        with pytest.raises(ValueError, match="cannot normalize"):
            layer_norm_rule(x, gamma, beta, 1e-5)
