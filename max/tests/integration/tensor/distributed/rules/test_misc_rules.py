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

"""Pure-metadata tests for miscellaneous placement rules."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.sharding import (
    DeviceMapping,
    TensorLayout,
)
from max.experimental.sharding.rules import (
    band_part_rule,
    fold_rule,
    irfft_rule,
    resize_rule,
)
from max.graph import Shape

from rules._fixtures import MESH_1D, M, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    """Build a TensorLayout from a DeviceMapping and shape."""
    return TensorLayout(dtype, Shape(shape), mapping)


class TestBandPartRule:
    def test_batch_sharded_ok(self) -> None:
        """S(0) is fine — only last 2 axes are forbidden."""
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(band_part_rule, layout)
        assert out.placements == (S(0),)

    def test_last_axis_auto_gathers(self) -> None:
        """Last-axis shard auto-gathers (band_part operates on last 2 axes jointly)."""
        layout = _layout(M(MESH_1D, S(2)), (4, 8, 3))
        _, (out,) = pick(band_part_rule, layout)
        assert out.placements == (R,)

    def test_second_last_axis_auto_gathers(self) -> None:
        """Second-last axis shard auto-gathers."""
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        _, (out,) = pick(band_part_rule, layout)
        assert out.placements == (R,)

    def test_replicated_ok(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8, 3))
        _, (out,) = pick(band_part_rule, layout)
        assert out.placements == (R,)


class TestFoldRule:
    def test_batch_sharded_ok(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(
            fold_rule, layout, output_size=(2, 4), kernel_size=(2, 2)
        )
        assert out.placements == (S(0),)

    def test_axis1_auto_gathers(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        _, (out,) = pick(
            fold_rule, layout, output_size=(2, 4), kernel_size=(2, 2)
        )
        assert out.placements == (R,)

    def test_axis2_auto_gathers(self) -> None:
        layout = _layout(M(MESH_1D, S(2)), (4, 8, 3))
        _, (out,) = pick(
            fold_rule, layout, output_size=(2, 4), kernel_size=(2, 2)
        )
        assert out.placements == (R,)


class TestResizeRule:
    def test_batch_only_ok(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8, 3))
        _, (out,) = pick(resize_rule, layout, shape=(4, 16, 6))
        assert out.placements == (S(0),)

    def test_non_batch_auto_gathers(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8, 3))
        _, (out,) = pick(resize_rule, layout, shape=(4, 16, 6))
        assert out.placements == (R,)


class TestIrfftRule:
    def test_batch_sharded_ok(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(irfft_rule, layout)
        assert out.placements == (S(0),)

    def test_last_axis_auto_gathers(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(irfft_rule, layout)
        assert out.placements == (R,)
