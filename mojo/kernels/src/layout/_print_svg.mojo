# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import Dict, Optional
from sys import sizeof
from layout import Layout, LayoutTensor
from layout.swizzle import Swizzle
from utils import Writer
from pathlib import Path


fn print_svg[
    dtype: DType,
    layout: Layout,
    layout_int_type: DType,
    linear_idx_type: DType,
    element_layout: Layout,
    masked: Bool, //,
    swizzle: Optional[Swizzle] = None,
    memory_bank: Optional[Tuple[Int, Int]] = None,
](
    tensor_base: LayoutTensor,
    tensors: List[
        LayoutTensor[
            dtype,
            layout,
            MutableAnyOrigin,
            element_layout=element_layout,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
            masked=masked,
        ]
    ],
    color_map: Optional[fn (Int, Int) -> String] = None,
    file_path: Optional[Path] = None,
) raises:
    var s = String()
    _print_svg_impl[swizzle, memory_bank](tensor_base, tensors, s, color_map)
    if file_path:
        file_path.value().write_text(s)
    else:
        print(s)


fn _print_svg_impl[
    dtype: DType,
    layout: Layout,
    layout_int_type: DType,
    linear_idx_type: DType,
    element_layout: Layout,
    masked: Bool,
    W: Writer, //,
    swizzle: Optional[Swizzle] = None,
    memory_bank: Optional[Tuple[Int, Int]] = None,
](
    tensor_base: LayoutTensor,
    tensors: List[
        LayoutTensor[
            dtype,
            layout,
            MutableAnyOrigin,
            element_layout=element_layout,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
            masked=masked,
        ]
    ],
    mut writer: W,
    color_map: Optional[fn (Int, Int) -> String] = None,
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
        StaticString("#FFFFFF"),
        StaticString("#93C572"),
        StaticString("#ECFFDC"),
    )

    var cell_size = 80
    var margin = 40
    var text_margin = 30
    var width = (tensor_base.layout[1].size() + 2) * cell_size + 2 * margin
    var height = (tensor_base.layout[0].size() + 2) * cell_size + 2 * margin

    writer.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    writer.write(
        '<svg width="',
        width,
        '" height="',
        height,
        '" xmlns="http://www.w3.org/2000/svg">\n',
    )

    # Print layout legends
    writer.write(
        '<text x="',
        margin,
        '" y="',
        margin + 20,
        '" fill="',
        colors[0],
        '" opacity="0.4">Layout: ',
        tensor_base.layout,
        " ",
        tensor_base.element_layout,
        "</text>\n",
    )
    if len(tensors) > 0:
        writer.write(
            '<text x="',
            margin,
            '" y="',
            margin + 40,
            '" fill="',
            colors[1],
            '" opacity="0.4">Layout: ',
            tensors[0].layout,
            " ",
            tensors[0].element_layout,
            "</text>\n",
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
            writer.write(
                '<rect x="',
                x,
                '" y="',
                y,
                '" width="',
                cell_size,
                '" height="',
                cell_size,
                '" fill="',
            )
            if color_map and swizzle:
                writer.write(color_map.value()(idx, 0))
            else:
                writer.write(colors[0])
            writer.write('" opacity="0.2" stroke="black"/>\n')
            writer.write(
                '<text font-size="small" x="',
                x + cell_size / 2,
                '" y="',
                y + cell_size / 2 + 20,
                '" dominant-baseline="middle" text-anchor="middle">',
                idx,
            )

            @parameter
            if memory_bank:
                writer.write(
                    " b=",
                    (
                        (idx * sizeof[tensor_base.dtype]())
                        // memory_bank.value()[0]
                    )
                    % memory_bank.value()[1],
                )
            writer.write("</text>\n")
            if swizzle:
                writer.write(
                    '<text font-size="x-small" fill="gainsboro" x="',
                    x + 10,
                    '" y="',
                    y + 15,
                    '" dominant-baseline="middle" text-anchor="middle">',
                    non_swizzled_idx,
                    "</text>\n",
                )

    fn draw_element(
        x: Int,
        y: Int,
        color: String,
        t: Int,
        element_idx: Int,
        mut writer: W,
    ):
        writer.write(
            '<rect x="',
            x,
            '" y="',
            y,
            '" width="',
            cell_size,
            '" height="',
            cell_size,
            '" fill="',
            color,
            '" opacity="0.2" stroke="black"/>\n'
            + '<text font-size="large" x="',
            x + cell_size / 2,
            '" y="',
            y + cell_size / 2,
            '" dominant-baseline="middle" text-anchor="middle">T',
            t,
            " V",
            element_idx,
            "</text>\n",
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
                            ) if color_map else String(
                                colors[orig_pos[0].value() % 2 + 1]
                            )
                            draw_element(x, y, color, t, element_idx, writer)
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
                    ) if color_map else String(
                        colors[orig_pos[0].value() % 2 + 1]
                    )
                    draw_element(x, y, color, t, element_idx, writer)
                    element_idx += 1

    # Draw row labels
    for i in range(tensor_base.layout[0].size()):
        var y = start_y + i * cell_size + cell_size / 2
        writer.write(
            '<text x="',
            margin,
            '" y="',
            y,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="monospace" font-size="larger">'
            ),
            i,
            "</text>\n",
        )

    # Draw column labels
    for j in range(tensor_base.layout[1].size()):
        var x = margin + text_margin + j * cell_size + cell_size / 2
        writer.write(
            '<text x="',
            x,
            '" y="',
            start_y - text_margin / 2,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="monospace" font-size="larger">'
            ),
            j,
            "</text>\n",
        )

    # SVG Footer
    writer.write("</svg>\n")
