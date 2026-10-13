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
from std.testing import assert_equal

from layout import Coord, Idx, TileTensor, col_major, coord, row_major
from layout.tile_layout import Layout as TileLayout
from layout._print_svg import print_svg
from layout.swizzle import Swizzle


def assert_fragment_addresses[
    element_layout: TileLayout = TileLayout(coord[1, 1], coord[0, 0]),
](
    tensor: TileTensor[mut=False, ...],
    base: TileTensor[mut=False, ...],
    expected: List[Int],
) raises:
    var offset = (Int(tensor.unsafe_ptr()) - Int(base.unsafe_ptr())) // size_of[
        Float32
    ]()
    var index = 0
    for row in range(Int(tensor.dim[0]())):
        for column in range(Int(tensor.dim[1]())):
            for erow in range(element_layout.static_shape[0]):
                for ecolumn in range(element_layout.static_shape[1]):
                    assert_equal(
                        Int(tensor.layout(Coord(row, column)))
                        + offset
                        + Int(element_layout(Coord(erow, ecolumn))),
                        expected[index],
                    )
                    index += 1
    assert_equal(index, len(expected))


def assert_svg_fingerprint(
    path: String, expected_size: Int, expected: UInt64
) raises:
    var svg = Path(path).read_text()
    var fingerprint: UInt64 = 14695981039346656037
    for byte in svg.as_bytes():
        fingerprint = (fingerprint ^ UInt64(byte)) * 1099511628211
    assert_equal(svg.byte_length(), expected_size)
    assert_equal(fingerprint, expected)


def test_svg_nvidia_shape() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    comptime tensor_dist_type = type_of(
        tensor.vectorize[1, 2]().distribute[row_major[8, 4]()](0).as_imm()
    )

    var tensor_list = List[tensor_dist_type]()
    for i in range(32):
        tensor_list.append(
            tensor.vectorize[1, 2]().distribute[row_major[8, 4]()](i).as_imm()
        )

    for lane in range(32):
        var expected = List[Int]()
        for row in range(2):
            for column in range(2):
                for element in range(2):
                    expected.append(
                        16 * (lane // 4)
                        + 2 * (lane % 4)
                        + 128 * row
                        + 8 * column
                        + element
                    )
        assert_fragment_addresses[TileLayout(coord[1, 2], coord[0, 1])](
            tensor_list[lane], tensor.as_imm(), expected
        )

    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t // 4])

    print_svg[element_layout=TileLayout(coord[1, 2], coord[0, 1])](
        tensor.as_imm(),
        tensor_list,
        color_map,
        file_path=Path("./test_svg_nvidia_shape.svg"),
    )


def test_svg_nvidia_tile() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    var expected = List[Int]()
    for row in range(4):
        for column in range(4):
            for erow in range(2):
                for ecolumn in range(2):
                    expected.append(
                        8 + 32 * row + 2 * column + 16 * erow + ecolumn
                    )
    assert_fragment_addresses[TileLayout(coord[2, 2], coord[16, 1])](
        tensor_dist.as_imm(), tensor.as_imm(), expected
    )
    print_svg[element_layout=TileLayout(coord[2, 2], coord[16, 1])](
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_nvidia_tile.svg"),
    )


def test_svg_nvidia_tile_memory_bank() raises:
    # nvidia tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    var tensor_dist = tensor.vectorize[2, 2]().tile[4, 4](0, 1)
    var expected = List[Int]()
    for row in range(4):
        for column in range(4):
            for erow in range(2):
                for ecolumn in range(2):
                    expected.append(
                        8 + 32 * row + 2 * column + 16 * erow + ecolumn
                    )
    assert_fragment_addresses[TileLayout(coord[2, 2], coord[16, 1])](
        tensor_dist.as_imm(), tensor.as_imm(), expected
    )
    print_svg[
        memory_bank=(4, 32),
        element_layout=TileLayout(coord[2, 2], coord[16, 1]),
    ](
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_nvidia_tile_memory_bank.svg"),
    )


def test_svg_amd_shape_a() raises:
    # amd tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    var tensor_dist = tensor.distribute[col_major[16, 4]()](0)
    assert_fragment_addresses(
        tensor_dist.as_imm(), tensor.as_imm(), [0, 4, 8, 12]
    )
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_amd_shape_a.svg"),
    )


def test_svg_amd_shape_b() raises:
    # amd tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    var tensor_dist = tensor.distribute[row_major[4, 16]()](0)
    assert_fragment_addresses(
        tensor_dist.as_imm(), tensor.as_imm(), [0, 64, 128, 192]
    )
    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm()],
        file_path=Path("./test_svg_amd_shape_b.svg"),
    )


def test_svg_amd_shape_d() raises:
    # amd tensor core a matrix fragment
    comptime layout = row_major[16, 16]()
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    var tensor_dist = tensor.vectorize[4, 1]().distribute[row_major[4, 16]()](
        10
    )
    var tensor_dist2 = tensor.vectorize[4, 1]().distribute[row_major[4, 16]()](
        11
    )
    assert_fragment_addresses[TileLayout(coord[4, 1], coord[16, 0])](
        tensor_dist.as_imm(), tensor.as_imm(), [10, 26, 42, 58]
    )
    assert_fragment_addresses[TileLayout(coord[4, 1], coord[16, 0])](
        tensor_dist2.as_imm(), tensor.as_imm(), [11, 27, 43, 59]
    )
    print_svg[element_layout=TileLayout(coord[4, 1], coord[16, 0])](
        tensor.as_imm(),
        [tensor_dist.as_imm(), tensor_dist2.as_imm()],
        file_path=Path("./test_svg_amd_shape_d.svg"),
    )


def test_svg_wgmma_shape() raises:
    # wgmma tensor core a matrix fragment
    comptime layout = TileLayout(
        Coord(coord[8, 8], coord[8, 2]),
        Coord(coord[8, 64], coord[1, 512]),
    )
    var stack = Array[Float32, layout.static_product](fill={})
    var tensor = TileTensor(stack, layout)
    comptime fragment_layout = TileLayout(
        Coord(Idx[8], coord[2, 2]), Coord(Idx[64], coord[4, 512])
    )
    var tensor_dist = TileTensor(tensor.unsafe_ptr(), fragment_layout)
    var tensor_dist2 = TileTensor(tensor.unsafe_ptr() + 24, fragment_layout)
    var expected = List[Int]()
    var expected2 = List[Int]()
    for row in range(8):
        for column in range(4):
            var address = 64 * row + 4 * (column % 2) + 512 * (column // 2)
            expected.append(address)
            expected2.append(address + 24)
    assert_fragment_addresses(tensor_dist.as_imm(), tensor.as_imm(), expected)
    assert_fragment_addresses(tensor_dist2.as_imm(), tensor.as_imm(), expected2)

    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("red"),
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("purple"),
            StaticString("orange"),
            StaticString("pink"),
            StaticString("brown"),
            StaticString("gray"),
            StaticString("black"),
            StaticString("white"),
        ]
        return String(colors[t])

    print_svg(
        tensor.as_imm(),
        [tensor_dist.as_imm(), tensor_dist2.as_imm()],
        color_map,
        file_path=Path("./test_svg_wgmma_shape.svg"),
    )


def test_svg_swizzle() raises:
    comptime layout = row_major[8, 8]()
    var stack = Array[Float32, layout.static_product](fill={})
    comptime swizzle = Swizzle(3, 0, 3)
    var tensor = TileTensor(stack, layout)

    # the figure generated here is identical to
    # https://docs.nvidia.com/cuda/parallel-thread-execution/_images/async-warpgroup-smem-layout-128B-k.png
    def color_map(t: Int, v: Int) -> String:
        var colors = [
            StaticString("blue"),
            StaticString("green"),
            StaticString("yellow"),
            StaticString("red"),
            StaticString("lightblue"),
            StaticString("lightgreen"),
            StaticString("lightyellow"),
            StaticString("salmon"),  # lighter variant of red
        ]
        return String(colors[t % len(colors)])

    for row in range(8):
        for column in range(8):
            var address = Int(tensor.layout(Coord(row, column)))
            assert_equal(address, 8 * row + column)
            assert_equal(swizzle(address), 8 * row + (column ^ row))
    print_svg[swizzle](
        tensor.as_imm(),
        List[type_of(tensor.as_imm())](),
        color_map=color_map,
        file_path=Path("./test_svg_swizzle.svg"),
    )


def main() raises:
    test_svg_nvidia_shape()
    test_svg_nvidia_tile()
    test_svg_nvidia_tile_memory_bank()
    test_svg_amd_shape_a()
    test_svg_amd_shape_b()
    test_svg_amd_shape_d()
    test_svg_wgmma_shape()
    test_svg_swizzle()

    # Complete legacy render fingerprints include addresses, labels, colors, and banks.
    assert_svg_fingerprint(
        "test_svg_nvidia_shape.svg", 199694, 13004575881472271883
    )
    assert_svg_fingerprint(
        "test_svg_nvidia_tile.svg", 123716, 13242220274835241
    )
    assert_svg_fingerprint(
        "test_svg_nvidia_tile_memory_bank.svg", 124916, 14839201142227891815
    )
    assert_svg_fingerprint(
        "test_svg_amd_shape_a.svg", 99779, 2479487580217036271
    )
    assert_svg_fingerprint(
        "test_svg_amd_shape_b.svg", 99776, 6735268042707884551
    )
    assert_svg_fingerprint(
        "test_svg_amd_shape_d.svg", 101366, 5087618133045518495
    )
    assert_svg_fingerprint(
        "test_svg_wgmma_shape.svg", 407618, 18286544078747828906
    )
    assert_svg_fingerprint("test_svg_swizzle.svg", 34001, 12680549840839863326)
