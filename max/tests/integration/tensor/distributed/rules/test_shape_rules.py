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

"""Pure-metadata tests for shape-manipulation placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping
from max.experimental.sharding.rules.shape import (
    argsort_rule,
    broadcast_to_rule,
    flatten_rule,
    gather_nd_rule,
    gather_rule,
    outer_rule,
    pad_rule,
    passthrough_rule,
    permute_rule,
    repeat_interleave_rule,
    reshape_rule,
    same_placement_multi_input_rule,
    slice_tensor_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    transpose_rule,
    unsqueeze_rule,
)
from max.experimental.sharding.types import TensorLayout
from max.graph import Shape, ShapeLike

from rules._fixtures import MESH_1D, MESH_2, MESH_2D, M, R, S


def _layout(
    mapping: DeviceMapping, shape: ShapeLike, dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Passthrough
# ═════════════════════════════════════════════════════════════════════════


class TestPassthrough:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = passthrough_rule(layout)
        assert out.to_placements() == (R,)

    def test_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = passthrough_rule(layout)
        assert out.to_placements() == (S(0),)


# ═════════════════════════════════════════════════════════════════════════
#  Permute / Transpose
# ═════════════════════════════════════════════════════════════════════════


class TestPermute:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8, 3))
        _, (out,) = permute_rule(layout, dims=[2, 0, 1])
        assert out.to_placements() == (R,)

    def test_sharded_axis0_moves(self) -> None:
        """S(0) with dims=[1,0,2] -> S(1) (axis 0 goes to position 1)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = permute_rule(layout, dims=[1, 0, 2])
        assert out.to_placements() == (S(1),)

    def test_sharded_axis2_moves(self) -> None:
        """S(2) with dims=[2,0,1] -> S(0) (axis 2 goes to position 0)."""
        layout = _layout(M(MESH_1D, S(2)), (4, 8, 3))
        _, (out,) = permute_rule(layout, dims=[2, 0, 1])
        assert out.to_placements() == (S(0),)

    def test_identity_permute(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        _, (out,) = permute_rule(layout, dims=[0, 1, 2])
        assert out.to_placements() == (S(1),)


class TestTranspose:
    def test_swap_sharded_axis(self) -> None:
        """S(0) + transpose(0,1) -> S(1)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = transpose_rule(layout, axis_1=0, axis_2=1)
        assert out.to_placements() == (S(1),)

    def test_swap_other_axis(self) -> None:
        """S(0) + transpose(1,2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = transpose_rule(layout, axis_1=1, axis_2=2)
        assert out.to_placements() == (S(0),)

    def test_negative_axes(self) -> None:
        """S(0) + transpose(-2,-1) on 3-D -> S(0) (swaps axes 1,2)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = transpose_rule(layout, axis_1=-2, axis_2=-1)
        assert out.to_placements() == (S(0),)


# ═════════════════════════════════════════════════════════════════════════
#  Unsqueeze / Squeeze
# ═════════════════════════════════════════════════════════════════════════


class TestUnsqueeze:
    def test_insert_before_sharded(self) -> None:
        """S(1) + unsqueeze(0) -> S(2) (shifted up)."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = unsqueeze_rule(layout, axis=0)
        assert out.to_placements() == (S(2),)

    def test_insert_after_sharded(self) -> None:
        """S(0) + unsqueeze(2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = unsqueeze_rule(layout, axis=2)
        assert out.to_placements() == (S(0),)

    def test_insert_at_sharded(self) -> None:
        """S(1) + unsqueeze(1) -> S(2) (at insertion point, shifts up)."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = unsqueeze_rule(layout, axis=1)
        assert out.to_placements() == (S(2),)


class TestSqueeze:
    def test_squeeze_non_sharded(self) -> None:
        """S(0) + squeeze(2) -> S(0) (unaffected, axes after shift down)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 1))
        _, (out,) = squeeze_rule(layout, axis=2)
        assert out.to_placements() == (S(0),)

    def test_squeeze_shifts_down(self) -> None:
        """S(2) + squeeze(0) -> S(1) (shifted down)."""
        layout = _layout(M(MESH_1D, S(2)), (1, 4, 8))
        _, (out,) = squeeze_rule(layout, axis=0)
        assert out.to_placements() == (S(1),)

    def test_squeeze_sharded_axis_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 1, 8))
        with pytest.raises(ValueError, match="squeeze"):
            squeeze_rule(layout, axis=1)


# ═════════════════════════════════════════════════════════════════════════
#  Flatten
# ═════════════════════════════════════════════════════════════════════════


class TestFlatten:
    def test_flatten_non_sharded_range(self) -> None:
        """S(0) + flatten(1, 2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        _, (out,) = flatten_rule(layout, start_dim=1, end_dim=2)
        assert out.to_placements() == (S(0),)

    def test_flatten_shifts_axis(self) -> None:
        """S(3) + flatten(0, 1) -> S(2) (shifted down by 1 removed dim)."""
        layout = _layout(M(MESH_1D, S(3)), (2, 4, 8, 3))
        _, (out,) = flatten_rule(layout, start_dim=0, end_dim=1)
        assert out.to_placements() == (S(2),)

    def test_flatten_across_sharded_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (2, 4, 8))
        with pytest.raises(ValueError, match="flatten"):
            flatten_rule(layout, start_dim=0, end_dim=2)

    def test_flatten_all_sharded_raises(self) -> None:
        """flatten(0, -1) with any sharding -> error."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        with pytest.raises(ValueError, match="flatten"):
            flatten_rule(layout, start_dim=0, end_dim=-1)


# ═════════════════════════════════════════════════════════════════════════
#  Reshape
# ═════════════════════════════════════════════════════════════════════════


class TestReshape:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(2, 16))
        assert out.to_placements() == (R,)
        assert isinstance(target_shape, Shape)
        assert target_shape == [2, 16]

    def test_sharded_maps_cleanly(self) -> None:
        """S(0) on [4, 8] -> reshape [4, 2, 4]: axis 0 maps 1:1."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(4, 2, 4))
        assert out.to_placements() == (S(0),)
        assert isinstance(target_shape, Shape)
        # Axis 0 is sharded across 4 devices: 4/4 = 1.
        assert target_shape == [1, 2, 4]

    def test_sharded_axis_merged_ok(self) -> None:
        """S(0) on [4, 8] -> reshape [32]: axis 0 boundary aligns, 32 % 4 = 0."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(32,))
        assert out.to_placements() == (S(0),)
        # Output is sharded on axis 0: 32/4 = 8.
        assert isinstance(target_shape, Shape)
        assert target_shape == [8]

    def test_shape_positional(self) -> None:
        """Accepts shape as positional arg."""
        layout = _layout(M(MESH_1D, R), (4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, (2, 16))
        assert out.to_placements() == (R,)
        assert isinstance(target_shape, Shape)
        assert target_shape == [2, 16]

    def test_sharded_split_dim_raises(self) -> None:
        """S(1) on [4, 8] -> reshape [4, 2, 4]: pure split into [2, 4],
        but neither candidate satisfies the leading-1 + divisible-by-N
        conditions for N=4. Sharding cannot land on either new axis."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        with pytest.raises(ValueError, match="no candidate has all preceding"):
            reshape_rule(layout, shape=(4, 2, 4))

    def test_split_unsharded_dim(self) -> None:
        """S(1) on [4, 8] -> reshape [2, 2, 8]: results in S(2)"""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(2, 2, 8))
        assert out.to_placements() == (S(2),)
        # Output is sharded on axis 2: 8/4 = 2.
        assert isinstance(target_shape, Shape)
        assert target_shape == [2, 2, 2]

    # ── Dynamic-dim cases ────────────────────────────────────────────

    def test_replicated_with_dynamic_dim(self) -> None:
        """Replicated on ('batch', 4, 8) -> ('batch', 32): passes through."""
        layout = _layout(M(MESH_1D, R), ("batch", 4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 32))
        assert out.to_placements() == (R,)
        # Replicated: each shard sees the full target shape.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 32]

    def test_sharded_dynamic_axis_identity(self) -> None:
        """S(0) on ('batch', 4) -> ('batch', 4): identity, S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4))
        # TODO(MXF-278): add support for symbolic dimension sharding
        with pytest.raises(ValueError, match="Symbolic dimension sharding"):
            reshape_rule(layout, shape=("batch", 4))

    def test_sharded_static_with_dynamic_passthrough(self) -> None:
        """S(1) on ('batch', 8) -> ('batch', 4, 2): static axis 1 splits cleanly, S(1) preserved."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 8, 8))
        (_, target_shape), (out,) = reshape_rule(
            layout, shape=("batch", 8, 4, 2)
        )
        assert out.to_placements() == (S(1),)
        # Output is sharded on axis 1: 8/4 = 2.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 2, 4, 2]

    def test_sharded_dynamic_axis_with_static_merge(self) -> None:
        """S(0) on ('batch', 4, 8) -> ('batch', 32): S(0) on 'batch' stays at 0."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        # TODO(MXF-278): add support for symbolic dimension sharding
        with pytest.raises(ValueError, match="Symbolic dimension sharding"):
            reshape_rule(layout, shape=("batch", 32))

    def test_sharded_static_merged_after_dynamic(self) -> None:
        """S(1) on ('batch', 4, 8) -> ('batch', 32): old static axis 1 merges into new axis 1."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 32))
        assert out.to_placements() == (S(1),)
        # Output is sharded on axis 1: 32/4 = 8.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 8]

    def test_sharded_trailing_static_merged_after_dynamic(self) -> None:
        """S(2) on ('batch', 4, 8) -> ('batch', 32): old static axis 2 merges into new axis 1."""
        layout = _layout(M(MESH_1D, S(2)), ("batch", 4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 32))
        assert out.to_placements() == (S(1),)
        # Output is sharded on axis 1: 32/4 = 8.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 8]

    def test_minus_one_absorbs_static_dynamic_sharded_unaffected(self) -> None:
        """S(0) on ('batch', 4, 8) -> ('batch', -1): -1 absorbs static dims, S(0) on 'batch' stays."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        # TODO(MXF-278): add support for symbolic dimension sharding
        with pytest.raises(ValueError, match="Symbolic dimension sharding"):
            reshape_rule(layout, shape=("batch", -1))

    def test_minus_one_absorbs_sharded_static(self) -> None:
        """S(1) on ('batch', 4, 8) -> ('batch', -1): -1 absorbs old axes 1+2 into new axis 1."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", -1))
        assert out.to_placements() == (S(1),)
        # Output is sharded on the inferred axis: -1 // 4 == -1 (still inferred per-shard).
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", -1]

    def test_split_static_dynamic_sharded_unaffected(self) -> None:
        """S(0) on ('batch', 32) -> ('batch', 4, 8): only static axis splits, S(0) on 'batch' stays."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 32))
        # TODO(MXF-278): add support for symbolic dimension sharding
        with pytest.raises(ValueError, match="Symbolic dimension sharding"):
            reshape_rule(layout, shape=("batch", 4, 8))

    def test_dynamic_dim_merged(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8, "dynamic"))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 4, -1))
        assert out.to_placements() == (S(1),)
        # Output is sharded on axis 1: 4/4 = 1.
        # The last axis should still be "-1", and computed by the single-device
        # reshape kernel.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 1, -1]

    def test_dynamic_dim_all_merged(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8, "dynamic"))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(-1,))
        assert out.to_placements() == (S(0),)
        assert isinstance(target_shape, Shape)
        assert target_shape == [-1]

    def test_split_sharded_static_with_dynamic_present(self) -> None:
        """S(1) on ('batch', 32) -> ('batch', 4, 8): pure split of the
        sharded static axis — sharding lands on the leftmost new axis
        whose size is divisible by the mesh size (4 % 4 == 0)."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 32))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 4, 8))
        assert out.to_placements() == (S(1),)
        # Output is sharded on new axis 1: 4/4 = 1.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 1, 8]

    def test_split_sharded_axis_lands_on_first_divisible(self) -> None:
        """Pure split, leftmost candidate (size 8) is divisible by N=2.
        Mirrors the user's (T, 2048) Sharded(1) -> (-1, 8, 256) case."""
        layout = _layout(M(MESH_2, S(1)), ("total_seq_len", 2048))
        (_, target_shape), (out,) = reshape_rule(layout, shape=(-1, 8, 256))
        assert out.to_placements() == (S(1),)
        # New axis 1 (size 8) is sharded across 2 devices: 8/2 = 4.
        # The -1 stays as -1 in the per-shard shape (resolved at runtime
        # by the local reshape kernel).
        assert isinstance(target_shape, Shape)
        assert target_shape == [-1, 4, 256]

    def test_split_sharded_axis_mixed_split_with_dynamic_raises(self) -> None:
        """Mixed split: the leftmost spanned new axis is a -1 that also
        absorbs the dynamic dim from another old axis. Mirrors the user's
        (T, 2048) Sharded(1) -> (-1, 4, 256) case (N=2). Local reshape
        per shard would interleave shard data, so reject."""
        layout = _layout(M(MESH_2, S(1)), ("total_seq_len", 2048))
        with pytest.raises(
            ValueError, match="absorb other axes' contributions"
        ):
            reshape_rule(layout, shape=(-1, 4, 256))

    def test_dynamic_dim_renamed_raises(self) -> None:
        """S(0) on ('batch', 4) -> ('seq', 4): dynamic dims must match by name."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4))
        # TODO(MXF-278): add support for symbolic dimension sharding
        with pytest.raises(ValueError, match="Dynamic dimensions in new shape"):
            reshape_rule(layout, shape=("seq", 4))

    def test_static_product_mismatch_with_dynamic_raises(self) -> None:
        """S(0) on ('batch', 4) -> ('batch', 8): static products must match."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4))
        with pytest.raises(ValueError, match="Static dimensions in new shape"):
            reshape_rule(layout, shape=("batch", 8))

    def test_multiple_minus_ones_raises(self) -> None:
        """('batch', 4, 8) -> ('batch', -1, -1): only one -1 dimension allowed."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        with pytest.raises(ValueError, match=r"-1"):
            reshape_rule(layout, shape=("batch", -1, -1))

    def test_new_dynamic_dim_not_in_old_raises(self) -> None:
        """S(0) on (4, 8) -> ('batch', 32): cannot introduce a new symbolic dim."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match=r"[Dd]ynamic dimensions"):
            reshape_rule(layout, shape=("batch", 32))

    def test_2d_mesh_with_dynamic_dim(self) -> None:
        """M(MESH_2D, R, S(1)) on ('batch', 4, 8) -> ('batch', 32):
        R stays R; S(1) on static axis 1 merges into new axis 1."""
        layout = _layout(M(MESH_2D, R, S(1)), ("batch", 4, 8))
        (_, target_shape), (out,) = reshape_rule(layout, shape=("batch", 32))
        assert out.to_placements() == (R, S(1))
        # MESH_2D has shape (2, 2). Mesh axis 0 is Replicated (no division);
        # mesh axis 1 is Sharded(1): 32/2 = 16.
        assert isinstance(target_shape, Shape)
        assert target_shape == ["batch", 16]


# ═════════════════════════════════════════════════════════════════════════
#  Broadcast_to / Split
# ═════════════════════════════════════════════════════════════════════════


class TestBroadcastTo:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (1, 8))
        _, (out,) = broadcast_to_rule(layout, shape=(4, 8))
        assert out.to_placements() == (R,)

    def test_sharded_passthrough(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 1))
        _, (out,) = broadcast_to_rule(layout, shape=(4, 8))
        assert out.to_placements() == (S(0),)


class TestSplit:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (8, 4))
        _, (out,) = split_rule(layout, split_sizes=[4, 4], axis=0)
        assert out.to_placements() == (R,)

    def test_sharded_non_split_axis(self) -> None:
        """S(0) + split on axis=1 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        _, (out,) = split_rule(layout, split_sizes=[2, 2], axis=1)
        assert out.to_placements() == (S(0),)

    def test_sharded_split_axis_divisible(self) -> None:
        """S(0) + split on axis=0: sizes must be divisible by shard count."""
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        _, (out,) = split_rule(layout, split_sizes=[4, 4], axis=0)
        assert out.to_placements() == (S(0),)

    def test_sharded_split_axis_not_divisible_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        with pytest.raises(ValueError, match="not evenly divisible"):
            split_rule(layout, split_sizes=[3, 5], axis=0)


# ═════════════════════════════════════════════════════════════════════════
#  Concat / Stack
# ═════════════════════════════════════════════════════════════════════════


class TestConcat:
    def test_same_placements(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = same_placement_multi_input_rule([a, b])
        assert out.to_placements() == (S(0),)

    def test_mixed_placements_raises(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        with pytest.raises(ValueError, match="same placements"):
            same_placement_multi_input_rule([a, b])


class TestStack:
    def test_shifts_axes(self) -> None:
        """S(0) + stack(axis=0) -> S(1) (new dim inserted at 0)."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = stack_rule([a, b], axis=0)
        assert out.to_placements() == (S(1),)

    def test_stack_at_end(self) -> None:
        """S(0) + stack(axis=2) -> S(0) (unaffected)."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = stack_rule([a, b], axis=2)
        assert out.to_placements() == (S(0),)

    def test_mixed_raises(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        with pytest.raises(ValueError, match="same placements"):
            stack_rule([a, b], axis=0)


# ═════════════════════════════════════════════════════════════════════════
#  Gather / Gather_nd
# ═════════════════════════════════════════════════════════════════════════


class TestGather:
    def test_non_sharded_axis(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (8, 4))
        indices = _layout(M(MESH_1D, R), (8, 2))
        _, (out,) = gather_rule(x, indices, axis=1)
        assert out.to_placements() == (S(0),)

    def test_sharded_axis_raises(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (8, 4))
        indices = _layout(M(MESH_1D, R), (4, 4))
        with pytest.raises(ValueError, match="gather"):
            gather_rule(x, indices, axis=0)


class TestGatherNd:
    def test_batch_dim_sharded_ok(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        indices = _layout(M(MESH_1D, R), (4, 8, 1))
        _, (out,) = gather_nd_rule(x, indices, batch_dims=2)
        assert out.to_placements() == (S(0),)

    def test_non_batch_dim_sharded_raises(self) -> None:
        x = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        indices = _layout(M(MESH_1D, R), (4, 2, 1))
        with pytest.raises(ValueError, match="non-batch axis"):
            gather_nd_rule(x, indices, batch_dims=1)


# ═════════════════════════════════════════════════════════════════════════
#  Pad / Slice / Repeat_interleave
# ═════════════════════════════════════════════════════════════════════════


class TestPad:
    def test_no_padding_on_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pad_rule(layout, paddings=[0, 0, 1, 1])
        assert out.to_placements() == (S(0),)

    def test_padding_on_sharded_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="pad"):
            pad_rule(layout, paddings=[1, 1, 0, 0])


class TestSliceTensor:
    def test_no_slice_on_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = slice_tensor_rule(
            layout, indices=[slice(None), slice(0, 4)]
        )
        assert out.to_placements() == (S(0),)

    def test_slice_on_sharded_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="slice"):
            slice_tensor_rule(layout, indices=[slice(0, 2), slice(None)])


class TestRepeatInterleave:
    def test_non_sharded_axis(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = repeat_interleave_rule(layout, axis=1)
        assert out.to_placements() == (S(0),)

    def test_sharded_axis_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="repeat_interleave"):
            repeat_interleave_rule(layout, axis=0)

    def test_axis_none_any_sharded_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError, match="axis=None"):
            repeat_interleave_rule(layout, axis=None)


# ═════════════════════════════════════════════════════════════════════════
#  Reject-all-sharded (argsort, nonzero, scatter_nd, etc.)
# ═════════════════════════════════════════════════════════════════════════


class TestRejectAllSharded:
    def test_replicated_ok(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = argsort_rule(layout)
        assert out.to_placements() == (R,)

    def test_any_sharded_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        with pytest.raises(ValueError):
            argsort_rule(layout)


# ═════════════════════════════════════════════════════════════════════════
#  Outer
# ═════════════════════════════════════════════════════════════════════════


class TestOuter:
    def test_both_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4,))
        y = _layout(M(MESH_1D, R), (6,))
        _, (out,) = outer_rule(x, y)
        assert out.to_placements() == (R,)

    def test_x_sharded(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4,))
        y = _layout(M(MESH_1D, R), (6,))
        _, (out,) = outer_rule(x, y)
        assert out.to_placements() == (S(0),)

    def test_y_sharded(self) -> None:
        x = _layout(M(MESH_1D, R), (4,))
        y = _layout(M(MESH_1D, S(0)), (6,))
        _, (out,) = outer_rule(x, y)
        assert out.to_placements() == (S(1),)

    def test_both_sharded_raises(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4,))
        y = _layout(M(MESH_1D, S(0)), (6,))
        with pytest.raises(ValueError, match="incompatible"):
            outer_rule(x, y)
