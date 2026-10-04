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
from std.pathlib import Path
from std.sys import size_of

from layout import Coord, TileTensor, TensorLayout, TensorEngine, coord
from layout.tile_layout import Layout as TileLayout
from layout.swizzle import Swizzle


def print_svg[
    tensor_list_origin: ImmOrigin,
    dtype: DType,
    layout: TensorLayout,
    engine: TensorEngine,
    linear_idx_type: DType,
    //,
    swizzle: Optional[Swizzle] = None,
    memory_bank: Optional[Tuple[Int, Int]] = None,
    element_layout: TileLayout = TileLayout(coord[1, 1], coord[0, 0]),
](
    tensor_base: TileTensor[mut=False, ...],
    tensors: List[
        TileTensor[
            dtype,
            layout,
            tensor_list_origin,
            Engine=engine,
            linear_idx_type=linear_idx_type,
        ]
    ],
    color_map: Optional[def(Int, Int) thin -> String] = None,
    file_path: Optional[Path] = None,
) raises:
    var s = String()
    _print_svg_impl[swizzle, memory_bank, element_layout](
        tensor_base, tensors, s, color_map
    )
    if file_path:
        file_path.value().write_text(s)
    else:
        print(s)


def _print_svg_impl[
    tensor_list_origin: ImmOrigin,
    dtype: DType,
    layout: TensorLayout,
    engine: TensorEngine,
    linear_idx_type: DType,
    W: Writer,
    //,
    swizzle: Optional[Swizzle] = None,
    memory_bank: Optional[Tuple[Int, Int]] = None,
    element_layout: TileLayout = TileLayout(coord[1, 1], coord[0, 0]),
](
    tensor_base: TileTensor[mut=False, ...],
    tensors: List[
        TileTensor[
            dtype,
            layout,
            tensor_list_origin,
            Engine=engine,
            linear_idx_type=linear_idx_type,
        ]
    ],
    mut writer: W,
    color_map: Optional[def(Int, Int) thin -> String] = None,
) raises:
    comptime assert tensor_base.rank == 2, "Layout rank must be 2"
    comptime assert layout.rank == 2, "Layout rank must be 2"
    comptime assert element_layout.rank == 2, "Element layout rank must be 2"
    comptime assert (
        element_layout.static_product == engine.element_size
    ), "Element layout must describe every SIMD lane"
    comptime assert (
        layout.all_dims_known and tensor_base.LayoutType.all_dims_known
    ), "SVG layouts must be static"
    if len(tensors) > 0:
        comptime assert (
            Coord[layout._shape_types[0]].static_product
            <= Coord[tensor_base.LayoutType._shape_types[0]].static_product
        ), "Base layout must have the largest first dimension"
        comptime assert (
            Coord[layout._shape_types[1]].static_product
            <= Coord[tensor_base.LayoutType._shape_types[1]].static_product
        ), "Base layout must have the largest second dimension"

    var colors: List[StaticString] = ["#FFFFFF", "#4A90E2", "#E8F0FF"]

    var cell_size = 80
    var margin = 40
    var text_margin = 30
    var width = (Int(tensor_base.dim[1]()) + 2) * cell_size + 2 * margin
    var height = (Int(tensor_base.dim[0]()) + 2) * cell_size + 2 * margin

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

    var map = Dict[Int, Tuple[Int, Int]]()
    var start_y = margin + 60  # Additional space for legends

    # Draw base layout
    for i in range(Int(tensor_base.dim[0]())):
        for j in range(Int(tensor_base.dim[1]())):
            var idx = Int(tensor_base.layout(Coord(i, j)))
            var non_swizzled_idx = idx

            comptime if swizzle:
                idx = swizzle.value()(idx)

            map[idx] = (i, j)
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
                Float64(x) + Float64(cell_size) / 2,
                '" y="',
                Float64(y) + Float64(cell_size) / 2 + 5,
                (
                    '" dominant-baseline="middle" text-anchor="middle"'
                    ' fill="#2C3E50">'
                ),
                idx,
            )

            comptime if memory_bank:
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

    def draw_element(
        x: Int,
        y: Int,
        color: String,
        t: Int,
        element_idx: Int,
        mut writer: W,
    ) {mut cell_size}:
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
            Float64(x) + Float64(cell_size) / 2,
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

    # SIMD width does not encode noncontiguous fragment-element strides.
    # Use the explicit element layout without loading the tensor's data.
    for t in range(len(tensors)):
        var tensor = tensors[t]
        var offset = (
            Int(tensor.unsafe_ptr()) - Int(tensor_base.unsafe_ptr())
        ) // size_of[Scalar[tensor.dtype]]()
        var element_idx = 0
        for i in range(Int(tensor.dim[0]())):
            for j in range(Int(tensor.dim[1]())):
                for e_i in range(element_layout.static_shape[0]):
                    for e_j in range(element_layout.static_shape[1]):
                        var idx = (
                            Int(tensor.layout(Coord(i, j)))
                            + offset
                            + Int(element_layout(Coord(e_i, e_j)))
                        )
                        var orig_pos = map[idx]
                        var x = margin + text_margin + orig_pos[1] * cell_size
                        var y = start_y + orig_pos[0] * cell_size
                        var color = color_map.value()(
                            t, element_idx
                        ) if color_map else String(colors[1])
                        draw_element(x, y, color, t, element_idx, writer)
                        element_idx += 1

    # Draw row labels with improved typography
    for i in range(Int(tensor_base.dim[0]())):
        var y = Float64(start_y + i * cell_size) + Float64(cell_size) / 2
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
    for j in range(Int(tensor_base.dim[1]())):
        var x = (
            Float64(margin + text_margin + j * cell_size)
            + Float64(cell_size) / 2
        )
        writer.write(
            '<text x="',
            x,
            '" y="',
            Float64(start_y) - Float64(text_margin) / 2,
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
