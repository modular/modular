# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import Dict, Optional
from sys import sizeof
from layout import Layout, LayoutTensor
from layout.swizzle import Swizzle


fn print_svg[
    dtype: DType,
    layout: Layout,
    rank: Int,
    element_layout: Layout,
    masked: Bool, //,
    swizzle: Optional[Swizzle] = None,
](
    tensor_base: LayoutTensor,
    tensors: List[
        LayoutTensor[
            dtype, layout, rank, element_layout=element_layout, masked=masked
        ]
    ],
    color_map: Optional[fn (Int, Int) -> String] = None,
    filename: Optional[String] = None,
) raises:
    # Given a base layout tensor and a sub tensor print the layouts
    # Verify rank constraint
    debug_assert(tensor_base.layout.rank() == 2, "Layout rank must be 2")

    if len(tensors) > 0:
        debug_assert(tensors[0].layout.rank() == 2, "Layout rank must be 2")

        debug_assert(
            tensors[0].layout[0].size() <= tensor_base.layout[0].size(),
            "Layout 0 should have the largest first dimension",
        )
        debug_assert(
            tensors[0].layout[1].size() <= tensor_base.layout[1].size(),
            "Layout 0 should have the largest second dimension",
        )

    var colors = List(
        "#FFFFFF",
        "#93C572",
        "#ECFFDC",
    )

    var cell_size = 80
    var margin = 40
    var text_margin = 30
    var width = (tensor_base.layout[1].size() + 2) * cell_size + 2 * margin
    var height = (tensor_base.layout[0].size() + 2) * cell_size + 2 * margin

    var svg = String('<?xml version="1.0" encoding="UTF-8"?>\n')
    svg += (
        '<svg width="'
        + String(width)
        + '" height="'
        + String(height)
        + '" xmlns="http://www.w3.org/2000/svg">\n'
    )

    # Print layout legends
    svg += (
        '<text x="'
        + String(margin)
        + '" y="'
        + String(margin + 20)
        + '" fill="'
        + colors[0]
        + '" opacity="0.4">Layout: '
        + String(tensor_base.layout)
        + " "
        + String(tensor_base.element_layout)
        + "</text>\n"
    )
    if len(tensors) > 0:
        svg += (
            '<text x="'
            + String(margin)
            + '" y="'
            + String(margin + 40)
            + '" fill="'
            + colors[1]
            + '" opacity="0.4">Layout: '
            + String(tensors[0].layout)
            + " "
            + String(tensors[0].element_layout)
            + "</text>\n"
        )

    var map = Dict[Int, IntTuple]()
    var start_y = margin + 60  # Additional space for legends

    # Draw base layout
    for i in range(tensor_base.layout[0].size()):
        for j in range(tensor_base.layout[1].size()):
            var idx = tensor_base.layout(IntTuple(i, j))
            var non_swizzled_idx = idx

            @parameter
            if swizzle:
                idx = swizzle.value()(idx)

            map[idx] = IntTuple(i, j)
            var x = margin + text_margin + j * cell_size
            var y = start_y + i * cell_size
            svg += (
                '<rect x="'
                + String(x)
                + '" y="'
                + String(y)
                + '" width="'
                + String(cell_size)
                + '" height="'
                + String(cell_size)
                + '" fill="'
            )
            if color_map and swizzle:
                svg += color_map.value()(idx, 0)
            else:
                svg += colors[0]
            svg += '" opacity="0.2" stroke="black"/>\n'
            svg += (
                '<text font-size="small" x="'
                + String(x + cell_size / 2)
                + '" y="'
                + String(y + cell_size / 2 + 20)
                + '" dominant-baseline="middle" text-anchor="middle">'
                + String(idx)
                + "</text>\n"
            )
            if swizzle:
                svg += (
                    '<text font-size="x-small" fill="gainsboro" x="'
                    + String(x + 10)
                    + '" y="'
                    + String(y + 15)
                    + '" dominant-baseline="middle" text-anchor="middle">'
                    + String(non_swizzled_idx)
                    + "</text>\n"
                )

    fn draw_element(
        x: Int, y: Int, color: String, t: Int, element_idx: Int
    ) -> String:
        return (
            '<rect x="'
            + String(x)
            + '" y="'
            + String(y)
            + '" width="'
            + String(cell_size)
            + '" height="'
            + String(cell_size)
            + '" fill="'
            + color
            + '" opacity="0.2" stroke="black"/>\n'
            + '<text font-size="large" x="'
            + String(x + cell_size / 2)
            + '" y="'
            + String(y + cell_size / 2)
            + '" dominant-baseline="middle" text-anchor="middle">T'
            + String(t)
            + " V"
            + String(element_idx)
            + "</text>\n"
        )

    for t in range(len(tensors)):
        var tensor = tensors[t]
        # Draw other layouts
        if tensor.element_layout.rank() == 2:
            var element_idx = 0
            for i in range(tensor.layout[0].size()):
                for j in range(tensor.layout[1].size()):
                    for e_i in range(tensor.element_layout[0].size()):
                        for e_j in range(tensor.element_layout[1].size()):
                            var offset = (
                                Int(tensor.ptr) - Int(tensor_base.ptr)
                            ) // sizeof[Scalar[tensor.dtype]]()
                            var element_offset = tensor.element_layout(
                                IntTuple(e_i, e_j)
                            )
                            var idx = tensor.layout(
                                IntTuple(i, j)
                            ) + offset + element_offset
                            var orig_pos = map[idx]
                            var x = margin + text_margin + orig_pos[
                                1
                            ].value() * cell_size
                            var y = start_y + orig_pos[0].value() * cell_size
                            var color = color_map.value()(
                                t, element_idx
                            ) if color_map else colors[
                                orig_pos[0].value() % 2 + 1
                            ]
                            svg += draw_element(x, y, color, t, element_idx)
                            element_idx += 1
        else:
            var element_idx = 0
            for i in range(tensor.layout[0].size()):
                for j in range(tensor.layout[1].size()):
                    var offset = (
                        Int(tensor.ptr) - Int(tensor_base.ptr)
                    ) // sizeof[Scalar[tensor.dtype]]()
                    var idx = tensor.layout(IntTuple(i, j)) + offset
                    var orig_pos = map[idx]
                    var x = margin + text_margin + orig_pos[
                        1
                    ].value() * cell_size
                    var y = start_y + orig_pos[0].value() * cell_size
                    var color = color_map.value()(
                        t, element_idx
                    ) if color_map else colors[orig_pos[0].value() % 2 + 1]
                    svg += draw_element(x, y, color, t, element_idx)
                    element_idx += 1

    # Draw row labels
    for i in range(tensor_base.layout[0].size()):
        var y = start_y + i * cell_size + cell_size / 2
        svg += (
            '<text x="'
            + String(margin)
            + '" y="'
            + String(y)
            + '" dominant-baseline="middle" text-anchor="middle"'
            ' font-family="monospace" font-size="larger">'
            + String(i)
            + "</text>\n"
        )

    # Draw column labels
    for j in range(tensor_base.layout[1].size()):
        var x = margin + text_margin + j * cell_size + cell_size / 2
        svg += (
            '<text x="'
            + String(x)
            + '" y="'
            + String(start_y - text_margin / 2)
            + '" dominant-baseline="middle" text-anchor="middle"'
            ' font-family="monospace" font-size="larger">'
            + String(j)
            + "</text>\n"
        )

    # SVG Footer
    svg += "</svg>\n"
    if filename:
        with open(filename.value(), "w") as f:
            f.write(svg)
    else:
        print(svg)
