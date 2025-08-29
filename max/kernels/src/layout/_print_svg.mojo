# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from pathlib import Path
from sys import size_of

from layout import Layout, LayoutTensor
from layout.swizzle import Swizzle


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

    var colors = List[StaticString]("#FFFFFF", "#4A90E2", "#E8F0FF")

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
    # Add white background
    writer.write(
        '<rect width="100%" height="100%" fill="white"/>\n',
    )
    # Define enhanced shadow filters
    writer.write(
        "<defs>\n",
        (
            '  <filter id="cellShadow" x="-20%" y="-20%" width="140%"'
            ' height="140%">\n'
        ),
        (
            '    <feDropShadow dx="2" dy="2" stdDeviation="3"'
            ' flood-color="#000000" flood-opacity="0.15"/>\n'
        ),
        "  </filter>\n",
        (
            '  <filter id="highlightShadow" x="-20%" y="-20%" width="140%"'
            ' height="140%">\n'
        ),
        (
            '    <feDropShadow dx="3" dy="3" stdDeviation="4" opacity="0.6"'
            ' flood-color="#2A5FC7" flood-opacity="0.3"/>\n'
        ),
        "  </filter>\n",
        "</defs>\n",
    )

    var map = Dict[Int, IntTuple]()
    var start_y = margin + 60  # Additional space for legends

    # Draw base layout
    for i in range(tensor_base.layout[0].size()):
        for j in range(tensor_base.layout[1].size()):
            var idx = tensor_base.layout([i, j])
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
            writer.write(
                '" opacity="0.6" stroke="#E1E8ED" stroke-width="1"'
                ' filter="url(#cellShadow)"/>\n'
            )
            writer.write(
                (
                    '<text font-family="-apple-system, BlinkMacSystemFont,'
                    ' Segoe UI, Roboto, Arial, sans-serif" font-size="16"'
                    ' font-weight="600" x="'
                ),
                x + cell_size / 2,
                '" y="',
                y + cell_size / 2 + 5,
                (
                    '" dominant-baseline="middle" text-anchor="middle"'
                    ' fill="#2C3E50">'
                ),
                idx,
            )

            @parameter
            if memory_bank:
                writer.write(
                    " b=",
                    (
                        (idx * size_of[tensor_base.dtype]())
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
            '" opacity="0.6" stroke="#2A5FC7" stroke-width="2"'
            ' filter="url(#highlightShadow)"/>\n'
            + '<text font-family="-apple-system, BlinkMacSystemFont, Segoe UI,'
            ' Roboto, Arial, sans-serif" font-size="16"'
            ' font-weight="700" x="',
            x + cell_size / 2,
            '" y="',
            y + 15,
            (
                '" dominant-baseline="middle" text-anchor="middle" fill="white"'
                ' text-shadow="0 1px 2px rgba(0,0,0,0.7)">T'
            ),
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
                            ) // size_of[Scalar[tensor.dtype]]()
                            var element_offset = tensor.element_layout(
                                [e_i, e_j]
                            )
                            var idx = (
                                tensor.layout([i, j]) + offset + element_offset
                            )
                            var orig_pos = map[idx]
                            var x = (
                                margin
                                + text_margin
                                + orig_pos[1].value() * cell_size
                            )
                            var y = start_y + orig_pos[0].value() * cell_size
                            var color = color_map.value()(
                                t, element_idx
                            ) if color_map else String(colors[1])
                            draw_element(x, y, color, t, element_idx, writer)
                            element_idx += 1
        else:
            var element_idx = 0
            for i in range(tensor.layout[0].size()):
                for j in range(tensor.layout[1].size()):
                    var offset = (
                        Int(tensor.ptr) - Int(tensor_base.ptr)
                    ) // size_of[Scalar[tensor.dtype]]()
                    var idx = tensor.layout([i, j]) + offset
                    var orig_pos = map[idx]
                    var x = (
                        margin + text_margin + orig_pos[1].value() * cell_size
                    )
                    var y = start_y + orig_pos[0].value() * cell_size
                    var color = color_map.value()(
                        t, element_idx
                    ) if color_map else String(colors[1])
                    draw_element(x, y, color, t, element_idx, writer)
                    element_idx += 1

    # Draw row labels with improved typography
    for i in range(tensor_base.layout[0].size()):
        var y = start_y + i * cell_size + cell_size / 2
        writer.write(
            '<text x="',
            margin,
            '" y="',
            y,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="-apple-system, BlinkMacSystemFont, Segoe UI,'
                ' Roboto, Arial, sans-serif" font-size="20" font-weight="600"'
                ' fill="#34495E">'
            ),
            i,
            "</text>\n",
        )

    # Draw column labels with improved typography
    for j in range(tensor_base.layout[1].size()):
        var x = margin + text_margin + j * cell_size + cell_size / 2
        writer.write(
            '<text x="',
            x,
            '" y="',
            start_y - text_margin / 2,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="-apple-system, BlinkMacSystemFont, Segoe UI,'
                ' Roboto, Arial, sans-serif" font-size="20" font-weight="600"'
                ' fill="#34495E">'
            ),
            j,
            "</text>\n",
        )

    # SVG Footer
    writer.write("</svg>\n")
