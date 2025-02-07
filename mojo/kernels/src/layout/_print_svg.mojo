# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from layout import Layout
from collections import Dict
from sys import sizeof


fn print_svg[
    dtype: DType,
    layout: Layout,
    rank: Int,
    element_layout: Layout,
    masked: Bool,
](
    tensor_base: LayoutTensor,
    tensors: List[
        LayoutTensor[
            dtype, layout, rank, element_layout=element_layout, masked=masked
        ]
    ],
) raises:
    # Given a base layout tensor and a sub tensor print the layouts
    # Verify rank constraint
    debug_assert(tensor_base.layout.rank() == 2, "Layout rank must be 2")
    debug_assert(tensors[0].layout.rank() == 2, "Layout rank must be 2")

    debug_assert(
        tensors[0].layout.shape[0].value()
        <= tensor_base.layout.shape[0].value(),
        "Layout 0 should have the largest first dimension",
    )
    debug_assert(
        tensors[0].layout.shape[1].value()
        <= tensor_base.layout.shape[1].value(),
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
    var width = (
        tensor_base.layout.shape[1].value() + 2
    ) * cell_size + 2 * margin
    var height = (
        tensor_base.layout.shape[0].value() + 2
    ) * cell_size + 2 * margin

    # SVG Header
    print('<?xml version="1.0" encoding="UTF-8"?>')
    print(
        '<svg width="',
        width,
        '" height="',
        height,
        '" xmlns="http://www.w3.org/2000/svg">',
        sep="",
    )

    # Print layout legends
    print(
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
        "</text>",
        sep="",
    )

    print(
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
        "</text>",
        sep="",
    )

    var map = Dict[Int, IntTuple]()
    var start_y = margin + 60  # Additional space for legends

    # Draw base layout
    for i in range(tensor_base.layout.shape[0].value()):
        for j in range(tensor_base.layout.shape[1].value()):
            var idx = tensor_base.layout(IntTuple(i, j))
            map[idx] = IntTuple(i, j)
            var x = margin + text_margin + j * cell_size
            var y = start_y + i * cell_size
            print(
                '<rect x="',
                x,
                '" y="',
                y,
                '" width="',
                cell_size,
                '" height="',
                cell_size,
                '" fill="',
                colors[0],
                '" opacity="0.2" stroke="black"/>',
                sep="",
            )
            print(
                '<text font-size="small" x="',
                x + cell_size / 2,
                '" y="',
                y + cell_size / 2 + 20,
                '" dominant-baseline="middle" text-anchor="middle">',
                idx,
                "</text>",
                sep="",
            )

    for t in range(len(tensors)):
        var tensor = tensors[t]
        # Draw other layouts
        if tensor.element_layout.rank() == 2:
            var element_idx = 0
            for i in range(tensor.layout.shape[0].value()):
                for j in range(tensor.layout.shape[1].value()):
                    for e_i in range(tensor.element_layout.shape[0].value()):
                        for e_j in range(
                            tensor.element_layout.shape[1].value()
                        ):
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

                            print(
                                '<rect x="',
                                x,
                                '" y="',
                                y,
                                '" width="',
                                cell_size,
                                '" height="',
                                cell_size,
                                '" fill="',
                                colors[orig_pos[0].value() % 2 + 1],
                                '" opacity="0.2" stroke="black"/>',
                                sep="",
                            )
                            print(
                                '<text font-size="large" x="',
                                x + cell_size / 2,
                                '" y="',
                                y + cell_size / 2,
                                (
                                    '" dominant-baseline="middle"'
                                    ' text-anchor="middle">'
                                ),
                                "T",
                                t,
                                " V",
                                element_idx,
                                "</text>",
                                sep="",
                            )
                            element_idx += 1
        else:
            var element_idx = 0
            for i in range(tensor.layout.shape[0].value()):
                for j in range(tensor.layout.shape[1].value()):
                    var offset = (
                        Int(tensor.ptr) - Int(tensor_base.ptr)
                    ) // sizeof[Scalar[tensor.dtype]]()
                    var idx = tensor.layout(IntTuple(i, j)) + offset
                    var orig_pos = map[idx]
                    var x = margin + text_margin + orig_pos[
                        1
                    ].value() * cell_size
                    var y = start_y + orig_pos[0].value() * cell_size
                    print(
                        '<rect x="',
                        x,
                        '" y="',
                        y,
                        '" width="',
                        cell_size,
                        '" height="',
                        cell_size,
                        '" fill="',
                        colors[1],
                        '" opacity="0.2" stroke="black"/>',
                        sep="",
                    )
                    print(
                        '<text font-size="large" x="',
                        x + cell_size / 2,
                        '" y="',
                        y + cell_size / 2,
                        '" dominant-baseline="middle" text-anchor="middle">',
                        "T",
                        t,
                        " V",
                        element_idx,
                        "</text>",
                        sep="",
                    )
                    element_idx += 1

    # Draw row labels
    for i in range(tensor_base.layout.shape[0].value()):
        var y = start_y + i * cell_size + cell_size / 2
        print(
            '<text x="',
            margin,
            '" y="',
            y,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="monospace" font-size="larger">'
            ),
            i,
            "</text>",
            sep="",
        )

    # Draw column labels
    for j in range(tensor_base.layout.shape[1].value()):
        var x = margin + text_margin + j * cell_size + cell_size / 2
        print(
            '<text x="',
            x,
            '" y="',
            start_y - text_margin / 2,
            (
                '" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="monospace" font-size="larger">'
            ),
            j,
            "</text>",
            sep="",
        )

    # SVG Footer
    print("</svg>")
