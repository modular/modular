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

from rules._fixtures import MESH_1D, M, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


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
        _, (out,) = reshape_rule(layout, shape=(2, 16))
        assert out.to_placements() == (R,)

    def test_sharded_maps_cleanly(self) -> None:
        """S(0) on [4, 8] -> reshape [4, 2, 4]: axis 0 maps 1:1."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = reshape_rule(layout, shape=(4, 2, 4))
        assert out.to_placements() == (S(0),)

    def test_sharded_axis_merged_ok(self) -> None:
        """S(0) on [4, 8] -> reshape [32]: axis 0 boundary aligns, 32 % 4 = 0."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = reshape_rule(layout, shape=(32,))
        assert out.to_placements() == (S(0),)

    def test_shape_positional(self) -> None:
        """Accepts shape as positional arg."""
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = reshape_rule(layout, (2, 16))
        assert out.to_placements() == (R,)

    def test_sharded_split_dim_not_divisible(self) -> None:
        """S(1) on [4, 8] -> reshape [4, 2, 4]: new dim 1 (size 2) not divisible by 4 devices."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        with pytest.raises(ValueError, match="not evenly divisible"):
            reshape_rule(layout, shape=(4, 2, 4))


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
