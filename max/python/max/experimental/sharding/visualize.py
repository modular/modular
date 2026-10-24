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

"""Draws a device mesh and the slice of a tensor each device holds.

``visualize`` is the whole interface. It shows the same value twice, blocks
beside a tree, as SVG and HTML in a notebook and as Unicode text anywhere
else. It accepts a :class:`DeviceMesh`, a :class:`TensorLayout` or a tensor,
or a dict of them to draw side by side.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Protocol, TypeAlias

from max.graph import Dim, StaticDim

from .mappings import DeviceMapping
from .mesh import DeviceMesh
from .per_shard_dim import global_shape
from .placements import Partial, Placement, Unknown
from .types import TensorLayout

__all__ = ["visualize"]


class _HasLayout(Protocol):
    """Anything with a sharded layout, such as a tensor."""

    @property
    def layout(self) -> TensorLayout: ...


_Drawable: TypeAlias = DeviceMesh | TensorLayout | _HasLayout

_COLORS = (
    "#e6c58a",
    "#d2856a",
    "#a9b88c",
    "#8fa4b8",
    "#d7b04f",
    "#b59ab5",
    "#7f9d6e",
    "#7fa8a0",
)
_INK = "#2f2418"
# The blocks sit beside the tree, so they take about a third of a notebook
# cell rather than filling it.
_SIZE = 300.0
# Values drawn side by side share a row, so each is drawn smaller.
_SIDE_BY_SIDE_SIZE = 170.0
_TEXT_GAP = "   "

# A cell of the grid: its two text lines, the devices coloring it, and
# whether it is see-through, as a partial sum or a value of unknown relation
# is: what each device holds is not the tensor itself.
_Cell = tuple[str, str, Sequence[int], bool]


def _mark(placements: Sequence[Placement]) -> str:
    """What a node holding data under ``placements`` holds of the tensor."""
    if any(isinstance(p, Unknown) for p in placements):
        return "unknown"
    if any(isinstance(p, Partial) for p in placements):
        return "partial"
    return ""


def _layout(value: TensorLayout | _HasLayout) -> TensorLayout:
    if isinstance(value, TensorLayout):
        return value
    if isinstance(getattr(value, "layout", None), TensorLayout):
        return value.layout
    hint = (
        " Wrap it in a TensorLayout to give it a dtype and a shape."
        if isinstance(value, DeviceMapping)
        else ""
    )
    raise TypeError(
        "visualize expects a DeviceMesh, a TensorLayout or a Tensor, got "
        f"{type(value).__name__}.{hint}"
    )


def _chunks(layout: TensorLayout) -> tuple[list[list[int]], list[list[Dim]]]:
    """Per device the chunk it holds of each tensor axis, and each chunk's size.

    Mesh axes sharding the same tensor axis nest in mesh order, the first one
    outermost, as ``Sharded.local_dim`` lays the shards out.
    """
    mesh = layout.mesh
    chunk = [[0] * layout.rank for _ in range(mesh.num_devices)]
    count = [1] * layout.rank
    for m, p in enumerate(layout.placements):
        if (t := p.localized_axis()) is None:
            continue
        t %= layout.rank
        for d in range(mesh.num_devices):
            chunk[d][t] = chunk[d][t] * mesh.mesh_shape[m] + mesh.device_coord(
                d, m
            )
        count[t] *= mesh.mesh_shape[m]
    sizes = [[Dim(0)] * n for n in count]
    for d, local in enumerate(layout.local_types):
        for t, dim in enumerate(local.shape):
            sizes[t][chunk[d][t]] = dim
    return chunk, sizes


def _slice(sizes: Sequence[Dim], lo: int, hi: int) -> str:
    """NumPy slice text for chunks ``lo:hi`` of an axis split into ``sizes``."""
    start = "" if lo == 0 else str(sum(sizes[:lo], Dim(0)))
    stop = "" if hi == len(sizes) else str(sum(sizes[:hi], Dim(0)))
    return f"{start}:{stop}"


def _header(layout: TensorLayout, title: str | None) -> str:
    shape = ", ".join(str(d) for d in global_shape(layout.shape))
    return _titled(title, f"{layout.dtype} [{shape}] with {layout.mapping!r}")


def _titled(title: str | None, header: str) -> str:
    return f"{title}: {header}" if title else header


def _brief(value: _Drawable, title: str) -> str:
    """A header short enough for a value drawn beside others."""
    if isinstance(value, DeviceMesh):
        return _titled(title, repr(value))
    layout = _layout(value)
    shape = ", ".join(str(d) for d in global_shape(layout.shape))
    placements = ", ".join(repr(p) for p in layout.placements)
    return f"{title}: [{shape}] {placements}"


def _tree(
    mesh: DeviceMesh,
    devices: Sequence[int],
    label: Callable[[Sequence[int], int], str] | None,
    depth: int = 0,
    prefix: str = "",
) -> Iterator[str]:
    """One line per mesh node below the root, grouped by coordinate."""
    groups: dict[int, list[int]] = {}
    for d in devices:
        groups.setdefault(mesh.device_coord(d, depth), []).append(d)
    for i, (coord, members) in enumerate(sorted(groups.items())):
        last = i == len(groups) - 1
        node = f"{mesh.axis_names[depth]}={coord}"
        if depth + 1 == mesh.ndim:
            node += f"  device {members[0]}"
        if label is not None:
            node += f" - {label(members, depth + 1)}"
        yield f"{prefix}{'└── ' if last else '├── '}{node}"
        if depth + 1 < mesh.ndim:
            branch = prefix + ("    " if last else "│   ")
            yield from _tree(mesh, members, label, depth + 1, branch)


def _tree_lines(value: _Drawable, title: str | None) -> tuple[str, list[str]]:
    """The header and one line per node below it."""
    if isinstance(value, DeviceMesh):
        devices = range(value.num_devices)
        return _titled(title, repr(value)), list(_tree(value, devices, None))
    layout = _layout(value)
    chunk, sizes = _chunks(layout)

    def label(devices: Sequence[int], depth: int) -> str:
        slices = ", ".join(
            _slice(
                sizes[t],
                min(chunk[d][t] for d in devices),
                max(chunk[d][t] for d in devices) + 1,
            )
            for t in range(layout.rank)
        )
        prefix = {"unknown": "unknown ", "partial": "partial sum of ", "": ""}
        return f"{prefix[_mark(layout.placements[:depth])]}tensor[{slices}]"

    devices = range(layout.mesh.num_devices)
    return _header(layout, title), [
        label(devices, 0),
        *_tree(layout.mesh, devices, label),
    ]


def _grid(
    value: _Drawable,
) -> tuple[list[list[_Cell]], list[float], list[float]] | None:
    """The cells by row and the relative row and column sizes, or None.

    None means the blocks cannot show this value -- a mesh of more than two
    axes, or more than two sharded tensor axes -- and the tree stands alone.
    """
    if isinstance(value, DeviceMesh):
        if value.ndim > 2:
            return None
        cols = value.mesh_shape[-1]
        rows: list[list[_Cell]] = [
            [
                (
                    f"device {d}",
                    ", ".join(
                        f"{name}={value.device_coord(d, i)}"
                        for i, name in enumerate(value.axis_names)
                    ),
                    [d],
                    False,
                )
                for d in range(r * cols, (r + 1) * cols)
            ]
            for r in range(value.num_devices // cols)
        ]
        return rows, [1.0] * len(rows), [1.0] * cols

    layout = _layout(value)
    chunk, sizes = _chunks(layout)
    axes = (
        list(range(layout.rank))
        if layout.rank <= 2
        else [t for t, s in enumerate(sizes) if len(s) > 1]
    )
    if len(axes) > 2:
        return None
    padded: list[int | None] = [None, None]
    padded[2 - len(axes) :] = axes
    row_axis, col_axis = padded
    mark = _mark(layout.placements)

    def cell(r: int, c: int) -> _Cell:
        devices = [
            d
            for d in range(layout.mesh.num_devices)
            if (row_axis is None or chunk[d][row_axis] == r)
            and (col_axis is None or chunk[d][col_axis] == c)
        ]
        slices = ", ".join(
            _slice(sizes[t], chunk[devices[0]][t], chunk[devices[0]][t] + 1)
            for t in range(layout.rank)
        )
        return (
            f"device{'s' if len(devices) > 1 else ''} "
            + ",".join(str(d) for d in devices),
            f"[{slices}]" + (f" {mark}" if mark else ""),
            devices,
            bool(mark),
        )

    def weights(axis: int | None) -> list[float]:
        dims = sizes[axis] if axis is not None else [Dim(1)]
        if all(isinstance(d, StaticDim) for d in dims):
            return [float(int(d)) for d in dims]
        return [1.0] * len(dims)

    row_w, col_w = weights(row_axis), weights(col_axis)
    cells = [[cell(r, c) for c in range(len(col_w))] for r in range(len(row_w))]
    return cells, row_w, col_w


def _text(rows: Sequence[Sequence[_Cell]]) -> list[str]:
    width = (
        max(len(s) for row in rows for a, b, _, _ in row for s in (a, b)) + 2
    )
    rule = "─" * width

    def line(parts: Sequence[str], left: str, mid: str, right: str) -> str:
        return left + mid.join(parts) + right

    lines = [line([rule] * len(rows[0]), "┌", "┬", "┐")]
    for i, row in enumerate(rows):
        for texts in zip(*((a, b) for a, b, _, _ in row), strict=True):
            lines.append(line([t.center(width) for t in texts], "│", "│", "│"))
        last = i == len(rows) - 1
        edge = ("└", "┴", "┘") if last else ("├", "┼", "┤")
        lines.append(line([rule] * len(row), *edge))
    return lines


def _svg(
    rows: Sequence[Sequence[_Cell]],
    row_w: Sequence[float],
    col_w: Sequence[float],
    size: float = _SIZE,
) -> str:
    """Draws blocks in proportion to their size with a gap between them.

    A block held by several devices is one stripe per device, without gaps.
    The result is ``size`` wide at most, so it sits next to the tree rather
    than filling the cell on its own, and shrinks with a narrower container.
    """
    aspect = sum(row_w) / sum(col_w)
    width, height = (
        (size, size * aspect) if aspect <= 1 else (size / aspect, size)
    )
    height = max(height, 34.0 * len(rows))
    gap = 3.0

    def spans(
        weights: Sequence[float], total: float
    ) -> list[tuple[float, float]]:
        scale = (total - gap * (len(weights) - 1)) / sum(weights)
        edges, pos = [], 0.0
        for w in weights:
            edges.append((pos, pos + w * scale))
            pos += w * scale + gap
        return edges

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'style="width:{width:.0f}px;max-width:100%;height:auto;display:block"'
        f' font-size="9" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" fill="{_INK}">'
    ]
    for (y0, y1), row in zip(spans(row_w, height), rows, strict=True):
        for (x0, x1), (head, body, devices, see_through) in zip(
            spans(col_w, width), row, strict=True
        ):
            stripe = (x1 - x0) / len(devices)
            opacity = ' fill-opacity=".4"' if see_through else ""
            for i, d in enumerate(devices):
                parts.append(
                    f'<rect x="{x0 + i * stripe:.1f}" y="{y0:.1f}" '
                    f'width="{stripe:.1f}" height="{y1 - y0:.1f}" '
                    f'fill="{_COLORS[d % len(_COLORS)]}"{opacity}/>'
                )
            parts.append(
                f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
                f'height="{y1 - y0:.1f}" fill="none" stroke="{_INK}" '
                'stroke-opacity=".3"/>'
                f'<text x="{x0 + 4:.1f}" y="{y0 + 11:.1f}" font-weight="600">'
                f"{html.escape(head)}</text>"
                f'<text x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2 + 3:.1f}" '
                f'text-anchor="middle" font-weight="600">{html.escape(body)}'
                "</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _columns(left: Sequence[str], right: Sequence[str]) -> list[str]:
    """Pads ``left`` to a common width and sets ``right`` beside it."""
    width = max(len(line) for line in left)
    height = max(len(left), len(right))

    def at(lines: Sequence[str], i: int) -> str:
        return lines[i] if i < len(lines) else ""

    return [
        f"{at(left, i):<{width}}{_TEXT_GAP}{at(right, i)}".rstrip()
        for i in range(height)
    ]


def _panel(header: str, *panes: str) -> str:
    """The header above the panes, which wrap onto their own line if narrow."""
    cells = "".join(
        f'<div style="flex:0 1 auto;min-width:0">{pane}</div>' for pane in panes
    )
    return (
        '<div style="font-family:ui-sans-serif,system-ui,sans-serif;'
        'font-size:12px">'
        f'<div style="margin-bottom:6px;opacity:.85">{html.escape(header)}'
        "</div>"
        '<div style="display:flex;flex-wrap:wrap;gap:18px;'
        f'align-items:flex-start">{cells}</div></div>'
    )


def _pre(lines: Sequence[str]) -> str:
    return (
        '<pre style="margin:0;font:11px/1.35 ui-monospace,monospace">'
        f"{html.escape(chr(10).join(lines))}</pre>"
    )


def _show(text: str, html_text: str) -> None:
    """Displays ``html_text`` in a notebook, and prints ``text`` anywhere else.

    The rich path is best-effort: without IPython, outside a running kernel,
    or if the display call itself fails, the Unicode text is printed instead.
    A frontend that receives both and cannot render HTML shows the text, so
    every route ends at a drawing.
    """
    try:
        if importlib.util.find_spec("IPython") is None:
            raise ImportError
        ipython = importlib.import_module("IPython")
        if ipython.get_ipython() is None:
            raise RuntimeError
        ipython.display.display(
            {"text/plain": text, "text/html": html_text}, raw=True
        )
    except Exception:
        print(text)


def visualize(
    value: _Drawable | Mapping[str, _Drawable],
    *,
    title: str | None = None,
    blocks: bool = True,
    tree: bool = True,
) -> None:
    """Draws a mesh or a tensor's slices as blocks beside the tree they share.

    The blocks on the left give one cell per device for a mesh, and one cell
    per slice for a tensor, with the devices that share a slice listed
    together. Cells are sized in proportion to the local shape when every
    dimension is static, and equally otherwise. A ``Partial`` or ``Unknown``
    cell is faded and labelled: what each device holds is not a piece of the
    drawn tensor. The tree gives one level per mesh axis and the value held
    at every node. Both describe the same value, so they share one header.

    A dict draws its values side by side, smaller, each under its key, so
    that related values such as an op's inputs and output read in one row:

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental.sharding import (
            DeviceMapping,
            DeviceMesh,
            Partial,
            Sharded,
            TensorLayout,
            visualize,
        )

        mesh = DeviceMesh(tuple(CPU() for _ in range(4)), (4,), ("tp",))

        def layout(shape, placement):
            return TensorLayout(
                DType.float32, shape, DeviceMapping(mesh, (placement,))
            )

        lhs = layout([4, 8], Sharded(1))
        rhs = layout([8, 8], Sharded(0))
        out = layout([4, 8], Partial())
        visualize({"lhs": lhs, "rhs": rhs, "out": out}, tree=False)

    Args:
        value: A :class:`DeviceMesh`, a :class:`TensorLayout`, or anything
            with a ``layout`` property that is a ``TensorLayout`` (such as a
            :class:`~max.experimental.tensor.Tensor`), or a dict of them keyed
            by title. A bare :class:`DeviceMapping` is not accepted; wrap it
            in a :class:`TensorLayout` to give it a dtype and a shape.
        title: Text prepended to the header, such as the variable name. For
            a dict, a line above the whole row.
        blocks: Whether to draw the blocks.
        tree: Whether to draw the tree. The blocks cannot show a mesh of more
            than two axes or more than two sharded tensor axes, so such a
            value is drawn as the tree alone however these two are set.

    Raises:
        TypeError: If ``value`` is none of those.
        ValueError: If both ``blocks`` and ``tree`` are false, or if a dict
            is empty.
    """
    if not (blocks or tree):
        raise ValueError("visualize needs blocks, tree, or both.")
    if not isinstance(value, Mapping):
        lines, panel = _drawing(
            value, _header_of(value, title), blocks, tree, _SIZE
        )
        _show("\n".join(lines), panel)
        return
    drawings = [
        _drawing(v, _brief(v, name), blocks, tree, _SIDE_BY_SIDE_SIZE)
        for name, v in value.items()
    ]
    if not drawings:
        raise ValueError("visualize needs at least one value to draw.")
    text = [title] if title else []
    lines = drawings[0][0]
    for more, _ in drawings[1:]:
        lines = _columns(lines, more)
    panels = "".join(
        f'<div style="flex:0 0 auto">{panel}</div>' for _, panel in drawings
    )
    _show(
        "\n".join([*text, *lines]),
        '<div style="font-family:ui-sans-serif,system-ui,sans-serif;'
        'font-size:12px">'
        + (
            f'<div style="margin-bottom:6px;font-weight:600">'
            f"{html.escape(title)}</div>"
            if title
            else ""
        )
        + '<div style="display:flex;flex-wrap:wrap;gap:24px;'
        f'align-items:flex-start">{panels}</div></div>',
    )


def _header_of(value: _Drawable, title: str | None) -> str:
    if isinstance(value, DeviceMesh):
        return _titled(title, repr(value))
    return _header(_layout(value), title)


def _drawing(
    value: _Drawable, header: str, blocks: bool, tree: bool, size: float
) -> tuple[list[str], str]:
    """The text lines and the HTML panel of one value under ``header``."""
    _, tree_lines = _tree_lines(value, None)
    grid = _grid(value) if blocks else None
    if grid is None:
        return [header, *tree_lines], _panel(header, _pre(tree_lines))
    rows, row_w, col_w = grid
    svg = _svg(rows, row_w, col_w, size)
    if not tree:
        return [header, *_text(rows)], _panel(header, svg)
    return (
        [header, *_columns(_text(rows), tree_lines)],
        _panel(header, svg, _pre(tree_lines)),
    )
