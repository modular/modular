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

"""Placement rules for ``conv2d``, ``conv3d``, and ``conv2d_transpose``."""

from __future__ import annotations

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.type import ConvInputLayout, FilterLayout

from ..action import AxisAssignment
from ..cost import P, R


def _conv_rows(
    spatial: int,
    transpose: bool,
    x: TensorLayout,
    filter: TensorLayout,
    groups: int,
    input_layout: ConvInputLayout,
    filter_layout: FilterLayout,
    bias: TensorLayout | None,
) -> list[AxisAssignment]:
    expected_filter = FilterLayout.RSCF if spatial == 2 else FilterLayout.QRSCF
    if filter_layout != expected_filter:
        raise ValueError(
            f"Unsupported filter layout for conv{spatial}d: {filter_layout}"
        )

    if input_layout == ConvInputLayout.NHWC:
        n_axis_x = 0
        cin_axis_x = x.rank - 1
        cout_axis_out = x.rank - 1
    elif input_layout == ConvInputLayout.NCHW:
        n_axis_x = 0
        cin_axis_x = 1
        cout_axis_out = 1
    else:
        raise ValueError(
            f"Unsupported input layout for conv{spatial}d: {input_layout}"
        )

    if transpose:
        cout_axis_filt = filter.rank - 2
        cin_axis_filt = filter.rank - 1
    else:
        cin_axis_filt = filter.rank - 2
        cout_axis_filt = filter.rank - 1

    rows = [
        AxisAssignment((R, R), (R,)),
        AxisAssignment((Sharded(n_axis_x), R), (Sharded(n_axis_x),)),
        AxisAssignment((R, Sharded(cout_axis_filt)), (Sharded(cout_axis_out),)),
    ]
    if groups == 1:
        rows.append(
            AxisAssignment((Sharded(cin_axis_x), Sharded(cin_axis_filt)), (P,))
        )
    rows.extend([AxisAssignment((P, R), (P,)), AxisAssignment((R, P), (P,))])
    if bias is not None:
        # The bias follows the output channels. Each device adds it, so a
        # partial-sum row would add it once per device.
        rows = [
            AxisAssignment(
                row.needed_inputs
                + (
                    Sharded(0)
                    if row.outputs[0] == Sharded(cout_axis_out)
                    else R,
                ),
                row.outputs,
            )
            for row in rows
            if row.outputs != (P,)
        ]
    return rows


def conv2d_rule(
    x: TensorLayout,
    filter: TensorLayout,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    bias: TensorLayout | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.RSCF,
) -> list[AxisAssignment]:
    """Strategies for ``conv2d``: data-parallel on N + channel-parallel on C."""
    return _conv_rows(
        2,
        False,
        x,
        filter,
        groups,
        input_layout,
        filter_layout,
        bias,
    )


def conv3d_rule(
    x: TensorLayout,
    filter: TensorLayout,
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    groups: int = 1,
    bias: TensorLayout | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.QRSCF,
) -> list[AxisAssignment]:
    """Strategies for ``conv3d``: same DP/channel-TP family as ``conv2d``."""
    return _conv_rows(
        3,
        False,
        x,
        filter,
        groups,
        input_layout,
        filter_layout,
        bias,
    )


def conv2d_transpose_rule(
    x: TensorLayout,
    filter: TensorLayout,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    output_paddings: tuple[int, int] = (0, 0),
    bias: TensorLayout | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.RSCF,
) -> list[AxisAssignment]:
    """Strategies for ``conv2d_transpose``: transpose-conv variant of ``conv2d``."""
    return _conv_rows(
        2,
        True,
        x,
        filter,
        1,
        input_layout,
        filter_layout,
        bias,
    )
