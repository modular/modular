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

"""Pure-metadata tests for convolution placement rules."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.experimental.sharding import DeviceMapping, Partial
from max.experimental.sharding.rules.conv import (
    conv2d_rule,
    conv2d_transpose_rule,
    conv3d_rule,
)
from max.experimental.sharding.types import TensorLayout

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, shape, mapping)


# NHWC layout: batch=0, H=1, W=2, C_in=3
# RSCF filter:  R=0, S=1, C_in=2, C_out=3
# Input shape: [N, H, W, C_in], Filter shape: [R, S, C_in, C_out]


class TestConv2dRule:
    """Tests for conv2d_rule with NHWC input + RSCF filter."""

    def test_both_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        """S(batch=0) on input, R on filter -> S(batch=0)."""
        x = _layout(M(MESH_1D, S(0)), (4, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (S(0),)

    def test_output_channel_tp(self) -> None:
        """R on input, S(C_out=3) on filter -> S(C_out=3) on output."""
        x = _layout(M(MESH_1D, R), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, S(3)), (3, 3, 3, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (S(3),)

    def test_input_channel_split_produces_partial(self) -> None:
        """S(C_in=3) on input x S(C_in=2) on filter -> Partial."""
        x = _layout(M(MESH_1D, S(3)), (1, 8, 8, 4))
        w = _layout(M(MESH_1D, S(2)), (3, 3, 4, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (Partial(),)

    def test_spatial_sharded_raises(self) -> None:
        """Sharding input on H (axis 1) -> error."""
        x = _layout(M(MESH_1D, S(1)), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        with pytest.raises(ValueError, match="spatial"):
            conv2d_rule(x, w)

    def test_spatial_w_sharded_raises(self) -> None:
        """Sharding input on W (axis 2) -> error."""
        x = _layout(M(MESH_1D, S(2)), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        with pytest.raises(ValueError, match="spatial"):
            conv2d_rule(x, w)

    def test_bilinear_partial_input(self) -> None:
        """P on input x R on filter -> P."""
        x = _layout(M(MESH_1D, P), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (P,)

    def test_cin_input_replicated_filter_raises(self) -> None:
        """S(C_in) on input but R on filter -> unsupported."""
        x = _layout(M(MESH_1D, S(3)), (1, 8, 8, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        with pytest.raises(NotImplementedError):
            conv2d_rule(x, w)

    def test_2d_mesh_dp_plus_channel_tp(self) -> None:
        """dp=S(batch=0), tp=S(C_out=3): DP + channel TP."""
        x = _layout(M(MESH_2D, S(0), R), (4, 8, 8, 3))
        w = _layout(M(MESH_2D, R, S(3)), (3, 3, 3, 16))
        _, (out,) = conv2d_rule(x, w)
        assert out.to_placements() == (S(0), S(3))


class TestConv3dRule:
    """Tests for conv3d_rule with NHWC-like 3D layout."""

    def test_both_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (1, 4, 4, 4, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 3, 16))
        _, (out,) = conv3d_rule(x, w)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4, 4, 4, 4, 3))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 3, 16))
        _, (out,) = conv3d_rule(x, w)
        assert out.to_placements() == (S(0),)

    def test_spatial_sharded_raises(self) -> None:
        """Sharding on any spatial axis (1, 2, 3) -> error."""
        w = _layout(M(MESH_1D, R), (3, 3, 3, 3, 16))
        for spatial_ax in [1, 2, 3]:
            x = _layout(M(MESH_1D, S(spatial_ax)), (1, 4, 4, 4, 3))
            with pytest.raises(ValueError, match="spatial"):
                conv3d_rule(x, w)


class TestConv2dTransposeRule:
    def test_both_replicated(self) -> None:
        x = _layout(M(MESH_1D, R), (1, 8, 8, 16))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        _, (out,) = conv2d_transpose_rule(x, w)
        assert out.to_placements() == (R,)

    def test_batch_sharded(self) -> None:
        x = _layout(M(MESH_1D, S(0)), (4, 8, 8, 16))
        w = _layout(M(MESH_1D, R), (3, 3, 3, 16))
        _, (out,) = conv2d_transpose_rule(x, w)
        assert out.to_placements() == (S(0),)
