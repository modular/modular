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

"""TensorLayout-based rules for convolution ops.

Convolution follows matmul-like semantics (bilinear in input x filter).
Partial handling: P x R = P (bilinear), P x non-R = resolve first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import TensorLayout
from max.graph.type import ConvInputLayout, FilterLayout

from ._common import RuleSignature, is_partial, is_replicated, is_sharded
from .matmul import _resolve_per_axis_placements


@dataclass(frozen=True)
class ConvRoles:
    """Axis role assignments for a convolution op."""

    x_batch: int
    x_cin: int
    x_spatial: frozenset[int]
    f_cin: int
    f_cout: int
    out_cout: int


def _conv_axis_rule(
    roles: ConvRoles,
) -> Callable[[Placement, Placement], tuple[Placement, Placement, Placement]]:
    def rule(
        pl: Placement, pr: Placement
    ) -> tuple[Placement, Placement, Placement]:
        if is_replicated(pl) and is_replicated(pr):
            return (Replicated(), Replicated(), Replicated())
        if is_sharded(pl, roles.x_batch) and is_replicated(pr):
            return (pl, pr, Sharded(roles.x_batch))
        if is_replicated(pl) and is_sharded(pr, roles.f_cout):
            return (pl, pr, Sharded(roles.out_cout))
        if is_sharded(pl, roles.x_cin) and is_sharded(pr, roles.f_cin):
            return (pl, pr, Partial())
        if (
            is_sharded(pl)
            and isinstance(pl, Sharded)
            and pl.axis in roles.x_spatial
        ):
            raise ValueError(
                f"conv: sharding input along spatial axis {pl.axis} "
                f"is not supported. Gather first."
            )
        if is_partial(pl) and is_replicated(pr):
            return (pl, pr, Partial(pl.reduce_op))
        if is_replicated(pl) and is_partial(pr):
            return (pl, pr, Partial(pr.reduce_op))
        if is_partial(pl) and is_partial(pr):
            raise ValueError(
                "conv: Partial x Partial is not supported. "
                "Use resolve_partials() on at least one input first."
            )
        if is_partial(pl):
            return (Replicated(), pr, _infer_output(Replicated(), pr))
        if is_partial(pr):
            return (pl, Replicated(), _infer_output(pl, Replicated()))
        raise NotImplementedError(f"conv: unsupported placements: {pl} x {pr}")

    def _infer_output(pl: Placement, pr: Placement) -> Placement:
        if is_replicated(pl) and is_replicated(pr):
            return Replicated()
        if is_replicated(pl) and is_sharded(pr, roles.f_cout):
            return Sharded(roles.out_cout)
        if is_sharded(pl, roles.x_batch) and is_replicated(pr):
            return Sharded(roles.x_batch)
        return Replicated()

    return rule


def _resolve_conv2d_roles(
    input_layout: ConvInputLayout, filter_layout: FilterLayout
) -> ConvRoles:
    if input_layout == ConvInputLayout.NHWC:
        x_batch, x_cin, x_spatial, out_cout = 0, 3, frozenset({1, 2}), 3
    elif input_layout == ConvInputLayout.NCHW:
        x_batch, x_cin, x_spatial, out_cout = 0, 1, frozenset({2, 3}), 1
    else:
        raise ValueError(f"Unsupported input layout: {input_layout}")
    if filter_layout == FilterLayout.RSCF:
        f_cin, f_cout = 2, 3
    else:
        raise ValueError(f"Unsupported filter layout: {filter_layout}")
    return ConvRoles(
        x_batch=x_batch,
        x_cin=x_cin,
        x_spatial=x_spatial,
        f_cin=f_cin,
        f_cout=f_cout,
        out_cout=out_cout,
    )


def _resolve_conv3d_roles(
    input_layout: ConvInputLayout, filter_layout: FilterLayout
) -> ConvRoles:
    if input_layout == ConvInputLayout.NHWC:
        x_batch, x_cin, x_spatial = 0, 4, frozenset({1, 2, 3})
    elif input_layout == ConvInputLayout.NCHW:
        x_batch, x_cin, x_spatial = 0, 1, frozenset({2, 3, 4})
    else:
        raise ValueError(f"Unsupported input layout: {input_layout}")
    if filter_layout == FilterLayout.QRSCF:
        f_cin, f_cout = 3, 4
    else:
        raise ValueError(f"Unsupported filter layout: {filter_layout}")
    return ConvRoles(
        x_batch=x_batch,
        x_cin=x_cin,
        x_spatial=x_spatial,
        f_cin=f_cin,
        f_cout=f_cout,
        out_cout=x_cin,
    )


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
) -> RuleSignature:
    """Sharding rule for conv2d."""
    roles = _resolve_conv2d_roles(input_layout, filter_layout)
    mesh = x.mapping.mesh

    suggested_x_p, suggested_w_p, out_p = _resolve_per_axis_placements(
        x.mapping.to_placements(),
        filter.mapping.to_placements(),
        mesh.ndim,
        _conv_axis_rule(roles),
        "conv2d",
    )
    xm = PlacementMapping(mesh, suggested_x_p)
    fm = PlacementMapping(mesh, suggested_w_p)
    return (
        (
            xm,
            fm,
            stride,
            dilation,
            padding,
            groups,
            bias,
            input_layout,
            filter_layout,
        ),
        (PlacementMapping(mesh, out_p),),
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
) -> RuleSignature:
    """Sharding rule for conv3d."""
    roles = _resolve_conv3d_roles(input_layout, filter_layout)
    mesh = x.mapping.mesh

    suggested_x_p, suggested_w_p, out_p = _resolve_per_axis_placements(
        x.mapping.to_placements(),
        filter.mapping.to_placements(),
        mesh.ndim,
        _conv_axis_rule(roles),
        "conv3d",
    )
    xm = PlacementMapping(mesh, suggested_x_p)
    fm = PlacementMapping(mesh, suggested_w_p)
    return (
        (
            xm,
            fm,
            stride,
            dilation,
            padding,
            groups,
            bias,
            input_layout,
            filter_layout,
        ),
        (PlacementMapping(mesh, out_p),),
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
) -> RuleSignature:
    """Sharding rule for conv2d transpose."""
    roles = _resolve_conv2d_roles(input_layout, filter_layout)
    if filter_layout == FilterLayout.RSCF:
        roles = ConvRoles(
            x_batch=roles.x_batch,
            x_cin=roles.x_cin,
            x_spatial=roles.x_spatial,
            f_cin=3,
            f_cout=2,
            out_cout=roles.out_cout,
        )
    mesh = x.mapping.mesh

    suggested_x_p, suggested_w_p, out_p = _resolve_per_axis_placements(
        x.mapping.to_placements(),
        filter.mapping.to_placements(),
        mesh.ndim,
        _conv_axis_rule(roles),
        "conv2d_transpose",
    )
    xm = PlacementMapping(mesh, suggested_x_p)
    fm = PlacementMapping(mesh, suggested_w_p)
    return (
        (
            xm,
            fm,
            stride,
            dilation,
            padding,
            output_paddings,
            bias,
            input_layout,
            filter_layout,
        ),
        (PlacementMapping(mesh, out_p),),
    )
