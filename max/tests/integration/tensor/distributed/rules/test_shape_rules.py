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
from max.experimental.sharding import (
    DeviceMapping,
    ShardingError,
    TensorLayout,
)
from max.experimental.sharding.per_shard_dim import (
    global_shape,
    local_dim_at,
    local_shape_at,
    make_per_shard_dim,
)
from max.experimental.sharding.placements import (
    Placement,
    local_shard_shape_from_global,
)
from max.experimental.sharding.rules import (
    argsort_rule,
    broadcast_to_rule,
    chunk_rule,
    concat_rule,
    flatten_rule,
    gather_nd_rule,
    gather_rule,
    outer_rule,
    pad_rule,
    permute_rule,
    rebind_rule,
    repeat_interleave_rule,
    reshape_rule,
    slice_tensor_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unsqueeze_rule,
)
from max.experimental.sharding.rules.shape import (
    localize_reshape_target,
    localize_same_shape_target,
)
from max.graph import Dim, Shape, ShapeLike

from rules._fixtures import MESH_1D, MESH_2, MESH_2D, M, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: ShapeLike, dtype: DType = DType.float32
) -> TensorLayout:
    """Builds a :class:`TensorLayout` whose ``shape`` is what ``Tensor.shape``
    returns at runtime."""
    global_shape = Shape(shape)
    local_shapes = local_shard_shape_from_global(
        global_shape, mapping.mesh, mapping.placements
    )
    sharded = {p.localized_axis() for p in mapping.placements}
    return TensorLayout(
        dtype,
        [
            make_per_shard_dim(
                [local[axis] for local in local_shapes],
                global_dim=size,
                force_wrap=True,
            )
            if axis in sharded
            else size
            for axis, size in enumerate(global_shape)
        ],
        mapping,
    )


def _reshape(
    layout: TensorLayout, shape: ShapeLike
) -> tuple[Shape, tuple[Placement, ...]]:
    """Reshapes as ``F.reshape`` does: the global target is localized, or
    stays global when the shard cannot keep its place."""
    localized = localize_reshape_target(layout, shape)
    target = global_shape(Shape(shape)) if localized is None else localized
    _, (out,) = pick(reshape_rule, layout, shape=target)
    return target, out.placements


def _unwrap(target: Shape, rank: int = 0) -> list[Dim]:
    """The sizes device ``rank`` reshapes its shard to."""
    return list(local_shape_at(target, rank))


class TestExplicitShapeRules:
    def test_plain_shape_is_global(self) -> None:
        # The rule reads a plain target as global, valid once gathered.
        layout = _layout(M(MESH_2, S(0)), (7, 8))
        action = pick(reshape_rule, layout, shape=(7, 4, 2))
        assert action.outputs[0].placements == (R,)

    def test_localized_global_shape_keeps_shard(self) -> None:
        layout = _layout(M(MESH_2, S(0)), (7, 8))
        target = localize_reshape_target(layout, (7, 4, 2))
        assert target is not None
        assert [local_dim_at(target[0], d) for d in range(2)] == [
            Dim(4),
            Dim(3),
        ]
        action = pick(reshape_rule, layout, shape=target)
        assert action.outputs[0].placements == (S(0),)

    def test_explicit_rank_shapes(self) -> None:
        # Rows split 4 + 3 over the mesh axis: the shard stays on them.
        layout = _layout(M(MESH_2, S(0)), (7, 8))
        rows = make_per_shard_dim([Dim(4), Dim(3)], force_wrap=True)
        action = pick(reshape_rule, layout, shape=(rows, 4, 2))
        assert action.outputs[0].placements == (S(0),)

    def test_per_device_sizes_on_another_axis_do_not_keep_the_shard(
        self,
    ) -> None:
        # Sizes given per device on another axis do not say where the shard
        # goes, and they do not reshape the whole tensor.
        layout = _layout(M(MESH_2, S(0)), (7, 8))
        cols = make_per_shard_dim([Dim(4), Dim(4)], force_wrap=True)
        with pytest.raises(ShardingError):
            pick(reshape_rule, layout, shape=(7, cols, 2))

    def test_rebind_never_redistributes(self) -> None:
        layout = _layout(M(MESH_2D, S(0), R), ("batch", 8))
        action = pick(rebind_rule, layout, shape=layout.shape)
        assert action.inputs == (layout.mapping,)
        assert action.outputs[0].placements == (S(0), R)

    def test_chunk_and_top_k_exact_outputs(self) -> None:
        layout = _layout(M(MESH_2, S(0)), (8, 8))
        assert len(pick(chunk_rule, layout, chunks=4, axis=1).outputs) == 4
        assert len(pick(top_k_rule, layout, k=2, axis=1).outputs) == 2


# ═════════════════════════════════════════════════════════════════════════
#  Permute / Transpose
# ═════════════════════════════════════════════════════════════════════════


class TestPermute:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8, 3))
        _, (out,) = pick(permute_rule, layout, dims=[2, 0, 1])
        assert out.placements == (R,)

    def test_sharded_axis0_moves(self) -> None:
        """S(0) with dims=[1,0,2] -> S(1) (axis 0 goes to position 1)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(permute_rule, layout, dims=[1, 0, 2])
        assert out.placements == (S(1),)

    def test_sharded_axis2_moves(self) -> None:
        """S(2) with dims=[2,0,1] -> S(0) (axis 2 goes to position 0)."""
        layout = _layout(M(MESH_1D, S(2)), (4, 8, 3))
        _, (out,) = pick(permute_rule, layout, dims=[2, 0, 1])
        assert out.placements == (S(0),)

    def test_identity_permute(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        _, (out,) = pick(permute_rule, layout, dims=[0, 1, 2])
        assert out.placements == (S(1),)


class TestTranspose:
    def test_swap_sharded_axis(self) -> None:
        """S(0) + transpose(0,1) -> S(1)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(transpose_rule, layout, axis_1=0, axis_2=1)
        assert out.placements == (S(1),)

    def test_swap_other_axis(self) -> None:
        """S(0) + transpose(1,2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(transpose_rule, layout, axis_1=1, axis_2=2)
        assert out.placements == (S(0),)

    def test_negative_axes(self) -> None:
        """S(0) + transpose(-2,-1) on 3-D -> S(0) (swaps axes 1,2)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(transpose_rule, layout, axis_1=-2, axis_2=-1)
        assert out.placements == (S(0),)


# ═════════════════════════════════════════════════════════════════════════
#  Unsqueeze / Squeeze
# ═════════════════════════════════════════════════════════════════════════


class TestUnsqueeze:
    def test_insert_before_sharded(self) -> None:
        """S(1) + unsqueeze(0) -> S(2) (shifted up)."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(unsqueeze_rule, layout, axis=0)
        assert out.placements == (S(2),)

    def test_insert_after_sharded(self) -> None:
        """S(0) + unsqueeze(2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(unsqueeze_rule, layout, axis=2)
        assert out.placements == (S(0),)

    def test_insert_at_sharded(self) -> None:
        """S(1) + unsqueeze(1) -> S(2) (at insertion point, shifts up)."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(unsqueeze_rule, layout, axis=1)
        assert out.placements == (S(2),)


class TestSqueeze:
    def test_squeeze_non_sharded(self) -> None:
        """S(0) + squeeze(2) -> S(0) (unaffected, axes after shift down)."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 1))
        _, (out,) = pick(squeeze_rule, layout, axis=2)
        assert out.placements == (S(0),)

    def test_squeeze_shifts_down(self) -> None:
        """S(2) + squeeze(0) -> S(1) (shifted down)."""
        layout = _layout(M(MESH_1D, S(2)), (1, 4, 8))
        _, (out,) = pick(squeeze_rule, layout, axis=0)
        assert out.placements == (S(1),)

    def test_squeeze_sharded_axis_falls_back(self) -> None:
        """Sharding on the squeezed axis: cost model auto-gathers to R."""
        layout = _layout(M(MESH_1D, S(1)), (4, 1, 8))
        _, (out,) = pick(squeeze_rule, layout, axis=1)
        assert out.placements == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Flatten
# ═════════════════════════════════════════════════════════════════════════


class TestFlatten:
    def test_flatten_non_sharded_range(self) -> None:
        """S(0) + flatten(1, 2) -> S(0) (unaffected)."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        _, (out,) = pick(flatten_rule, layout, start_dim=1, end_dim=2)
        assert out.placements == (S(0),)

    def test_flatten_shifts_axis(self) -> None:
        """S(3) + flatten(0, 1) -> S(2) (shifted down by 1 removed dim)."""
        layout = _layout(M(MESH_1D, S(3)), (2, 4, 8, 3))
        _, (out,) = pick(flatten_rule, layout, start_dim=0, end_dim=1)
        assert out.placements == (S(2),)

    def test_flatten_across_sharded_falls_back(self) -> None:
        """Flattening across a sharded axis: cost model auto-gathers to R."""
        layout = _layout(M(MESH_1D, S(1)), (2, 4, 8))
        _, (out,) = pick(flatten_rule, layout, start_dim=0, end_dim=2)
        assert out.placements == (R,)

    def test_flatten_all_sharded_falls_back(self) -> None:
        """flatten(0, -1) with any sharding: cost model auto-gathers."""
        layout = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        _, (out,) = pick(flatten_rule, layout, start_dim=0, end_dim=-1)
        assert out.placements == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Reshape
# ═════════════════════════════════════════════════════════════════════════


class TestReshape:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        target, out = _reshape(layout, (2, 16))
        assert out == (R,)
        assert _unwrap(target) == [2, 16]

    def test_sharded_maps_cleanly(self) -> None:
        """S(0) on [4, 8] -> reshape [4, 2, 4]: axis 0 maps 1:1."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        target, out = _reshape(layout, (4, 2, 4))
        assert out == (S(0),)
        # Axis 0 is sharded across 4 devices: 4/4 = 1.
        assert _unwrap(target) == [1, 2, 4]

    def test_sharded_axis_merged_ok(self) -> None:
        """S(0) on [4, 8] -> reshape [32]: axis 0 boundary aligns, 32 % 4 = 0."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        target, out = _reshape(layout, (32,))
        assert out == (S(0),)
        # Output is sharded on axis 0: 32/4 = 8.
        assert _unwrap(target) == [8]

    def test_shape_positional(self) -> None:
        """Accepts shape as positional arg."""
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(reshape_rule, layout, (2, 16))
        assert out.placements == (R,)

    def test_sharded_split_no_auto_fallback(self) -> None:
        """S(1) on [4, 8] -> reshape [4, 2, 4]: source axis 1 splits into
        target axes of sizes 2 and 4, and 2 can't host a 4-rank shard, so the
        input is gathered to Replicated."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, out = _reshape(layout, (4, 2, 4))
        assert out == (R,)

    def test_split_unper_rank_dim(self) -> None:
        """S(1) on [4, 8] -> reshape [2, 2, 8]: results in S(2)"""
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        target, out = _reshape(layout, (2, 2, 8))
        assert out == (S(2),)
        # Output is sharded on axis 2: 8/4 = 2.
        assert _unwrap(target) == [2, 2, 2]

    # ── Dynamic-dim cases ────────────────────────────────────────────

    def test_replicated_with_dynamic_dim(self) -> None:
        """Replicated on ('batch', 4, 8) -> ('batch', 32): passes through."""
        layout = _layout(M(MESH_1D, R), ("batch", 4, 8))
        target, out = _reshape(layout, ("batch", 32))
        assert out == (R,)
        # Replicated: each shard sees the full target shape.
        assert _unwrap(target) == [Dim("batch"), 32]

    def test_sharded_dynamic_axis_identity(self) -> None:
        """S(0) on ('batch', 4) -> (wrapper, 4): identity, S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4))
        target, out = _reshape(layout, (layout.shape[0], 4))
        assert out == (S(0),)
        # Symbolic batch axis is sharded -> per-rank distinct names
        # (rank 0's name appended with mesh-axis 'tp' and coord 0).
        assert _unwrap(target) == [Dim("batch_tp_0"), 4]
        assert _unwrap(target, rank=1) == [Dim("batch_tp_1"), 4]

    def test_sharded_static_with_dynamic_passthrough(self) -> None:
        """S(1) on ('batch', 8) -> ('batch', 4, 2): static axis 1 splits cleanly, S(1) preserved."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 8, 8))
        target, out = _reshape(layout, ("batch", 8, 4, 2))
        assert out == (S(1),)
        # Output is sharded on axis 1: 8/4 = 2.
        assert _unwrap(target) == [Dim("batch"), 2, 4, 2]

    def test_sharded_dynamic_axis_with_static_merge(self) -> None:
        """S(0) on ('batch', 4, 8) -> (wrapper, 32): S(0) on 'batch' stays at 0."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        target, out = _reshape(layout, (layout.shape[0], 32))
        assert out == (S(0),)
        # Symbolic batch is sharded -> per-rank distinct names.
        assert _unwrap(target) == [Dim("batch_tp_0"), 32]
        assert _unwrap(target, rank=3) == [Dim("batch_tp_3"), 32]

    def test_sharded_static_merged_after_dynamic(self) -> None:
        """S(1) on ('batch', 4, 8) -> ('batch', 32): old static axis 1 merges into new axis 1."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8))
        target, out = _reshape(layout, ("batch", 32))
        assert out == (S(1),)
        # Output is sharded on axis 1: 32/4 = 8.
        assert _unwrap(target) == [Dim("batch"), 8]

    def test_sharded_trailing_static_routes_via_local_dim(self) -> None:
        """S(2) on ('batch', 4, 8) -> ('batch', 32): the sharded axis is not
        the first merged one, so each device's block would scatter across the
        merged axis; the input is gathered instead."""
        layout = _layout(M(MESH_1D, S(2)), ("batch", 4, 8))
        _, out = _reshape(layout, ("batch", 32))
        assert out == (R,)

    def test_minus_one_absorbs_static_dynamic_sharded_unaffected(self) -> None:
        """S(0) on ('batch', 4, 8) -> (wrapper, -1): each device's reshape
        infers the -1 from its own shard (32)."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        target, out = _reshape(layout, (layout.shape[0], -1))
        assert out == (S(0),)
        assert _unwrap(target) == [Dim("batch_tp_0"), -1]
        assert _unwrap(target, rank=2) == [Dim("batch_tp_2"), -1]

    def test_minus_one_absorbs_sharded_static(self) -> None:
        """S(1) on ('batch', 4, 8) -> ('batch', -1): -1 resolves against the
        per-rank source (batch, 1, 8) to 8 (the only leftover static factor)."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8))
        target, out = _reshape(layout, ("batch", -1))
        assert out == (S(1),)
        assert _unwrap(target) == [Dim("batch"), 8]

    def test_split_static_dynamic_sharded_unaffected(self) -> None:
        """S(0) on ('batch', 32) -> (wrapper, 4, 8): only static axis splits,
        S(0) on the wrapper stays."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 32))
        target, out = _reshape(layout, (layout.shape[0], 4, 8))
        assert out == (S(0),)
        assert _unwrap(target) == [Dim("batch_tp_0"), 4, 8]
        assert _unwrap(target, rank=2) == [Dim("batch_tp_2"), 4, 8]

    def test_dynamic_dim_merged(self) -> None:
        """-1 absorbs the leftover ``8 * dynamic`` after axis-1 split, inferred
        by each device's reshape."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8, "dynamic"))
        target, out = _reshape(layout, ("batch", 4, -1))
        assert out == (S(1),)
        assert _unwrap(target) == [Dim("batch"), 1, -1]

    def test_dynamic_dim_all_merged(self) -> None:
        """S(1) on ('batch', 4, 8, 'dynamic') -> (-1,): the sharded axis is
        not the first merged one, so the input is gathered."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 4, 8, "dynamic"))
        _, out = _reshape(layout, (-1,))
        assert out == (R,)

    def test_split_sharded_static_with_dynamic_present(self) -> None:
        """S(1) on ('batch', 32) -> ('batch', 4, 8): pure split of the
        sharded static axis — sharding lands on the leftmost new axis."""
        layout = _layout(M(MESH_1D, S(1)), ("batch", 32))
        target, out = _reshape(layout, ("batch", 4, 8))
        assert out == (S(1),)
        # Output is sharded on new axis 1: 4/4 = 1.
        assert _unwrap(target) == [Dim("batch"), 1, 8]

    def test_split_sharded_axis_lands_on_first_divisible(self) -> None:
        """Pure split, leftmost candidate (size 8) is divisible by N=2."""
        layout = _layout(M(MESH_2, S(1)), ("total_seq_len", 2048))
        target, out = _reshape(layout, (-1, 8, 256))
        assert out == (S(1),)
        assert _unwrap(target)[1:] == [4, 256]

    def test_split_sharded_axis_mixed_split_with_dynamic(
        self,
    ) -> None:
        """Mixed split behind a ``-1``: no target axis provably starts where
        the sharded one does, so the input is gathered."""
        layout = _layout(M(MESH_2, S(1)), ("total_seq_len", 2048))
        _, out = _reshape(layout, (-1, 4, 256))
        assert out == (R,)

    def test_multiple_minus_ones_raises(self) -> None:
        """('batch', 4, 8) -> ('batch', -1, -1): only one -1 dimension allowed."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 4, 8))
        with pytest.raises(
            ValueError, match=r"at most one -1 dimension is allowed"
        ):
            reshape_rule(layout, shape=("batch", -1, -1))

    def test_2d_mesh_with_dynamic_dim(self) -> None:
        """M(MESH_2D, R, S(1)) on ('batch', 4, 8) -> ('batch', 32): the
        mesh-axis-1 shard stays on the merged dim."""
        layout = _layout(M(MESH_2D, R, S(1)), ("batch", 4, 8))
        _, placements = _reshape(layout, ("batch", 32))
        assert placements[0] == R
        assert isinstance(placements[1], type(S(0)))


class TestReshapeDynamic:
    """Reshape with non-static dims — no ``int(Dim)`` coercion."""

    def test_symbolic_batch_preserved_in_place(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), ("batch", 8))
        _, out = _reshape(layout, (layout.shape[0], 8))
        assert out == (S(0),)

    def test_symbolic_batch_minus_one_at_sharded_position(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), ("batch", 8))
        _, out = _reshape(layout, (-1, 4, 2))
        assert out == (S(0),)

    def test_symbolic_non_sharded_preserved(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), ("batch", 8))
        _, out = _reshape(layout, ("batch", 8, 1))
        assert out == (S(1),)

    def test_symbolic_two_minus_ones_raises(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), ("batch", 8))
        with pytest.raises(
            ValueError, match="at most one -1 dimension is allowed"
        ):
            reshape_rule(layout, shape=(-1, -1, 2))

    def test_symbolic_replicated_path_untouched(self) -> None:
        layout = _layout(M(MESH_1D, R), ("batch", 8))
        _, out = _reshape(layout, ("batch", 4, 2))
        assert out == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Broadcast_to / Split
# ═════════════════════════════════════════════════════════════════════════


class TestBroadcastTo:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (1, 8))
        _, (out,) = pick(broadcast_to_rule, layout, shape=(4, 8))
        assert out.placements == (R,)

    def test_sharded_passthrough(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 1))
        _, (out,) = pick(
            broadcast_to_rule,
            layout,
            shape=localize_same_shape_target(layout, (4, 8)),
        )
        assert out.placements == (S(0),)

    def test_symbolic_target_preserves_sharding(self) -> None:
        """Sharded symbolic axis preserved via an explicit wrapper target."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 1))
        _, (out,) = pick(broadcast_to_rule, layout, shape=(layout.shape[0], 8))
        assert out.placements == (S(0),)

    def test_symbolic_mismatch_raises(self) -> None:
        layout = _layout(M(MESH_1D, R), ("seq", 8))
        with pytest.raises(ValueError, match="must be either 1 or equal"):
            broadcast_to_rule(layout, shape=("batch", 8))


class TestSplit:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (8, 4))
        _, outs = pick(split_rule, layout, split_sizes=[4, 4], axis=0)
        assert all(out.placements == (R,) for out in outs)

    def test_sharded_non_split_axis(self) -> None:
        """S(0) + split on axis=1 -> S(0) preserved."""
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        _, outs = pick(split_rule, layout, split_sizes=[2, 2], axis=1)
        assert all(out.placements == (S(0),) for out in outs)

    def test_sharded_split_axis_divisible(self) -> None:
        """S(0) + split on axis=0: the rule gathers the split axis; ``split``
        itself splits each device's piece when the sizes divide evenly."""
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        _, outs = pick(split_rule, layout, split_sizes=[4, 4], axis=0)
        assert all(out.placements == (R,) for out in outs)

    def test_sharded_split_axis_not_divisible_falls_back(self) -> None:
        """Non-divisible split sizes on a sharded axis: cost model auto-gathers."""
        layout = _layout(M(MESH_1D, S(0)), (8, 4))
        _, outs = pick(split_rule, layout, split_sizes=[3, 5], axis=0)
        assert all(out.placements == (R,) for out in outs)

    def test_split_symbolic_sizes_unsharded_axis(self) -> None:
        """Symbolic split sizes on an unsharded axis keep the shard."""
        layout = _layout(M(MESH_1D, S(0)), (8, Dim("k1") + Dim("k2")))
        _, outs = pick(
            split_rule, layout, split_sizes=[Dim("k1"), Dim("k2")], axis=1
        )
        assert all(out.placements == (S(0),) for out in outs)


# ═════════════════════════════════════════════════════════════════════════
#  Reshape edge cases — symbolic sharded axis split / merge
# ═════════════════════════════════════════════════════════════════════════


class TestReshapeSymbolicEdges:
    """Probe corner cases of per-rank distinct symbolic naming."""

    def test_symbolic_sharded_axis_reshape_to_self(self) -> None:
        """Reshape to identical shape on a symbolic-sharded axis must
        stay shape-identity per rank."""
        layout = _layout(M(MESH_1D, S(0)), ("batch", 8))
        target, out = _reshape(layout, (layout.shape[0], 8))
        assert out == (S(0),)
        # Each rank's "batch" carries its rank-specific name.
        assert _unwrap(target, rank=0) == [Dim("batch_tp_0"), 8]
        assert _unwrap(target, rank=2) == [Dim("batch_tp_2"), 8]

    def test_symbolic_unsharded_axis_unaffected(self) -> None:
        """Symbolic dim on an unsharded axis: every rank holds ``[seq, 8]``."""
        layout = _layout(M(MESH_1D, R), ("seq", 8))
        target, out = _reshape(layout, ("seq", 8))
        assert out == (R,)
        assert _unwrap(target) == [Dim("seq"), 8]

    def test_static_uneven_reshape_same_shape_uniform_returns_per_rank(
        self,
    ) -> None:
        """7 elements over 4 ranks (sizes [2,2,2,1]) reshape to itself —
        each rank reshapes to its actual local size."""
        layout = _layout(M(MESH_1D, S(0)), (7, 8))
        target, out = _reshape(layout, (7, 8))
        assert out == (S(0),)
        assert [int(_unwrap(target, r)[0]) for r in range(4)] == [2, 2, 2, 1]


# ═════════════════════════════════════════════════════════════════════════
#  Concat / Stack
# ═════════════════════════════════════════════════════════════════════════


class TestConcat:
    def test_same_placements(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(concat_rule, [a, b])
        assert out.placements == (S(0),)

    def test_mixed_placements_aligns_to_cheapest(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(concat_rule, [a, b])
        assert out.placements == (S(0),)


class TestStack:
    def test_shifts_axes(self) -> None:
        """S(0) + stack(axis=0) -> S(1) (new dim inserted at 0)."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(stack_rule, [a, b], axis=0)
        assert out.placements == (S(1),)

    def test_stack_at_end(self) -> None:
        """S(0) + stack(axis=2) -> S(0) (unaffected)."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(stack_rule, [a, b], axis=2)
        assert out.placements == (S(0),)

    def test_mixed_aligns_to_first(self) -> None:
        """Mixed (S(0), R): cost model picks (S(0),S(0))->S(1) -- R->S(0) is
        free since the rhs is replicated, so this is the cheapest plan."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(stack_rule, [a, b], axis=0)
        assert out.placements == (S(1),)


# ═════════════════════════════════════════════════════════════════════════
#  Gather / Gather_nd
# ═════════════════════════════════════════════════════════════════════════


class TestGather:
    def test_non_sharded_axis(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (8, 4))
        indices = _layout(M(MESH_1D, R), (8, 2))
        _, (out,) = pick(gather_rule, x, indices, axis=1)
        assert out.placements == (S(0),)

    def test_sharded_axis_falls_back_to_replicated(self) -> None:
        """Sharding the gather-axis: the rule deliberately omits the
        expert-parallel ``(Sharded(a_axis), R) -> Partial(SUM)`` row
        (it would silently produce wrong results when the caller
        hasn't masked indices per rank — see
        ``rules/README.md`` "Gather and scatter expert-parallel
        rows"). The only feasible pick is the universal ``(R, R) -> R``
        fallback, which makes the picker allgather the input first.
        Callers that genuinely want EP semantics override
        ``gather.actions`` with their own rule."""
        x = _layout(M(MESH_1D, S(0)), (8, 4))
        indices = _layout(M(MESH_1D, R), (4, 4))
        _, (out,) = pick(gather_rule, x, indices, axis=0)
        assert out.placements == (R,)


class TestGatherNd:
    def test_batch_dim_sharded_ok(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        indices = _layout(M(MESH_1D, R), (4, 8, 1))
        _, (out,) = pick(gather_nd_rule, x, indices, batch_dims=2)
        assert out.placements == (S(0),)

    def test_non_batch_dim_sharded_falls_back(self) -> None:
        """Sharding a non-batch axis on input: cost model auto-gathers to R."""
        x = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        indices = _layout(M(MESH_1D, R), (4, 2, 1))
        _, (out,) = pick(gather_nd_rule, x, indices, batch_dims=1)
        assert out.placements == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Pad / Slice / Repeat_interleave
# ═════════════════════════════════════════════════════════════════════════


class TestPad:
    def test_no_padding_on_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(pad_rule, layout, paddings=[0, 0, 1, 1])
        assert out.placements == (S(0),)

    def test_padding_on_sharded_falls_back(self) -> None:
        """Pad on a sharded axis: cost model auto-gathers that axis."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(pad_rule, layout, paddings=[1, 1, 0, 0])
        assert out.placements == (R,)


class TestSliceTensor:
    def test_no_slice_on_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(
            slice_tensor_rule, layout, indices=[slice(None), slice(0, 4)]
        )
        assert out.placements == (S(0),)

    def test_slice_on_sharded_falls_back(self) -> None:
        """Slicing a sharded axis: cost model auto-gathers that axis."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(
            slice_tensor_rule, layout, indices=[slice(0, 2), slice(None)]
        )
        assert out.placements == (R,)

    def test_newaxis_does_not_hide_a_sliced_axis(self) -> None:
        """``None`` consumes no input axis, so it must not shift the rest.

        Counting it as one leaves the sharded axis looking untouched, and
        every rank then slices its own rows instead of the global ones.
        """
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(
            slice_tensor_rule,
            layout,
            indices=[None, slice(None), slice(0, 4)],
        )
        assert out.placements == (R,)

    def test_newaxis_shifts_the_kept_axis(self) -> None:
        """``None`` inserts an output axis, so a kept shard moves right."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(
            slice_tensor_rule,
            layout,
            indices=[None, slice(None), slice(None)],
        )
        assert out.placements == (S(1),)

    def test_integer_index_squeezes_the_kept_axis(self) -> None:
        """An integer index drops an axis, so a kept shard moves left."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 16))
        _, (out,) = pick(slice_tensor_rule, layout, indices=[0])
        assert out.placements == (S(0),)


class TestRepeatInterleave:
    def test_non_sharded_axis(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(repeat_interleave_rule, layout, repeats=2, axis=1)
        assert out.placements == (S(0),)

    def test_sharded_axis_falls_back(self) -> None:
        """repeat_interleave on a sharded axis: cost model auto-gathers."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(repeat_interleave_rule, layout, repeats=2, axis=0)
        assert out.placements == (R,)

    def test_axis_none_any_sharded_falls_back(self) -> None:
        """repeat_interleave(axis=None) auto-gathers any sharding."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(repeat_interleave_rule, layout, repeats=2, axis=None)
        assert out.placements == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Reject-all-sharded (argsort, nonzero, scatter_nd, etc.)
# ═════════════════════════════════════════════════════════════════════════


class TestRejectAllSharded:
    def test_replicated_ok(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(argsort_rule, layout)
        assert out.placements == (R,)

    def test_any_sharded_falls_back(self) -> None:
        """argsort needs a global view: cost model auto-gathers any sharding."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(argsort_rule, layout)
        assert out.placements == (R,)


# ═════════════════════════════════════════════════════════════════════════
#  Outer
# ═════════════════════════════════════════════════════════════════════════


class TestOuter:
    def test_both_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (4,))
        y = _layout(M(MESH_1D, R), (6,))
        _, (out,) = pick(outer_rule, x, y)
        assert out.placements == (R,)

    def test_x_sharded(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4,))
        y = _layout(M(MESH_1D, R), (6,))
        _, (out,) = pick(outer_rule, x, y)
        assert out.placements == (S(0),)

    def test_y_sharded(self) -> None:
        x = _layout(M(MESH_1D, R), (4,))
        y = _layout(M(MESH_1D, S(0)), (6,))
        _, (out,) = pick(outer_rule, x, y)
        assert out.placements == (S(1),)

    def test_both_sharded_picks_cheaper(self) -> None:
        """Both inputs sharded: cost model picks (R, S(0)) -> S(1) (cheaper)."""
        x = _layout(M(MESH_1D, S(0)), (4,))
        y = _layout(M(MESH_1D, S(0)), (6,))
        _, (out,) = pick(outer_rule, x, y)
        assert out.placements == (S(1),)


class TestTile:
    """Tiling a sharded axis locally would interleave the copies wrongly."""

    def test_tiled_sharded_axis_gathers(self) -> None:
        layout = _layout(M(MESH_2, S(0)), (4, 8))
        _, (out,) = pick(tile_rule, layout, repeats=(2, 1))
        assert out.placements == (R,)

    def test_untiled_sharded_axis_stays(self) -> None:
        layout = _layout(M(MESH_2, S(0)), (4, 8))
        _, (out,) = pick(tile_rule, layout, repeats=(1, 2))
        assert out.placements == (S(0),)
