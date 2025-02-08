# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s
from layout import Layout
from layout._print_svg import print_svg
from layout.tensor_builder import LayoutTensorBuild as tb
from layout import LayoutTensor


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


fn main() raises:
    test_svg_nvidia_shape()
    test_svg_nvidia_tile()
    test_svg_amd_shape_a()
    test_svg_amd_shape_b()
    test_svg_amd_shape_d()
