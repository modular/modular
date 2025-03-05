# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s
from layout import IntTuple, Layout, LayoutTensor
from layout._print_svg import print_svg
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.swizzle import Swizzle


fn test_svg_nvidia_shape() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # nvidia tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.vectorize[1, 2]().distribute[
        Layout.row_major(8, 4)
    ](0)

    var tensor_list = List[__type_of(tensor_dist)]()
    for i in range(32):
        tensor_list.append(
            tensor.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](i)
        )

    fn color_map(t: Int, v: Int) -> String:
        colors = List(
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
            "black",
            "white",
        )
        return colors[t // 4]

    print_svg(tensor, tensor_list, color_map)


fn test_svg_nvidia_tile() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # nvidia tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg(tensor, List(tensor_dist))


fn test_svg_nvidia_tile_memory_bank() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # nvidia tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    print_svg[memory_bank= (4, 32)](tensor, List(tensor_dist))


fn test_svg_amd_shape_a() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # amd tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.distribute[Layout.col_major(16, 4)](0)
    print_svg(tensor, List(tensor_dist))


fn test_svg_amd_shape_b() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # amd tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.distribute[Layout.row_major(4, 16)](0)
    print_svg(tensor, List(tensor_dist))


fn test_svg_amd_shape_d() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # amd tensor core a matrix fragment
    var tensor = tb[DType.float32]().row_major[16, 16]().alloc()
    var tensor_dist = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](10)
    var tensor_dist2 = tensor.vectorize[4, 1]().distribute[
        Layout.row_major(4, 16)
    ](11)
    print_svg(tensor, List(tensor_dist, tensor_dist2))


fn test_svg_wgmma_shape() raises:
    # CHECK: <?xml version="1.0" encoding="UTF-8"?>
    # wgmma tensor core a matrix fragment
    alias layout = Layout(
        IntTuple(IntTuple(8, 8), IntTuple(8, 2)),
        IntTuple(IntTuple(8, 64), IntTuple(1, 512)),
    )

    var tensor = LayoutTensor[
        DType.float32, layout, MutableAnyOrigin
    ].stack_allocation()
    var tensor_dist = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](0)
    var tensor_dist2 = tensor.vectorize[1, 1]().distribute[
        Layout.col_major(8, 4)
    ](3)

    fn color_map(t: Int, v: Int) -> String:
        colors = List(
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
            "black",
            "white",
        )
        return colors[t]

    print_svg(
        tensor,
        List[__type_of(tensor_dist)](tensor_dist, tensor_dist2),
        color_map,
    )


fn test_svg_swizzle() raises:
    alias layout = Layout.row_major(8, 8)
    alias swizzle = Swizzle(3, 0, 3)
    var tensor = LayoutTensor[
        DType.float32, layout, MutableAnyOrigin
    ].stack_allocation()

    # the figure generated here is identical to
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/async-warpgroup-smem-layout-128B-k.png
    fn color_map(t: Int, v: Int) -> String:
        var colors = List(
            "blue",
            "green",
            "yellow",
            "red",
            "lightblue",
            "lightgreen",
            "lightyellow",
            "salmon",  # lighter variant of red
        )
        return colors[t % len(colors)]

    print_svg[swizzle](
        tensor,
        List[__type_of(tensor)](),
        color_map=color_map,
    )


fn main() raises:
    test_svg_nvidia_shape()
    test_svg_nvidia_tile()
    test_svg_nvidia_tile_memory_bank()
    test_svg_amd_shape_a()
    test_svg_amd_shape_b()
    test_svg_amd_shape_d()
    test_svg_wgmma_shape()
    test_svg_swizzle()
