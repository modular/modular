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

from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.tile import tile

from utils import IndexList

alias layout_unknown_1d = Layout.row_major(UNKNOWN_VALUE)
alias layout_unknown_2d = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
alias layout_unknown_3d = Layout.row_major(
    UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
)
alias layout_unknown_4d = Layout.row_major(
    UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
)


# CHECK-LABEL: test_tile_eg1
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg1() raises:
    print("== test_tile_eg1")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2)](input_stack)

    input[0, 0] = 0
    input[0, 1] = 1
    input[1, 0] = 2
    input[1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 16](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 4)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](4, 4)),
        ),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg2
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg2() raises:
    print("== test_tile_eg2")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2)](input_stack)

    input[0, 0] = 0
    input[0, 1] = 1
    input[1, 0] = 2
    input[1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 3
    repeats[1] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 6 * 4](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(6, 4)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](6, 4)),
        ),
    )

    print()
    for i in range(6):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg3
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg3() raises:
    print("== test_tile_eg3")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2)](input_stack)

    input[0, 0] = 0
    input[0, 1] = 1
    input[1, 0] = 2
    input[1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 6](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 6)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](4, 6)),
        ),
    )

    print()
    for i in range(4):
        for j in range(6):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg4
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
fn test_tile_eg4() raises:
    print("== test_tile_eg4")
    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2)](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 3](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(3)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 1
    repeats[2] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 2 * 2](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 2, 2)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_3d](
            input_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](2, 2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](3)),
        ),
        LayoutTensor[type, layout_unknown_3d](
            output_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](4, 2, 2)),
        ),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg5
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
fn test_tile_eg5() raises:
    print("== test_tile_eg5")
    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2)](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 3](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(3)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 1
    repeats[2] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 2 * 4](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 2, 4)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_3d](
            input_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](2, 2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](3)),
        ),
        LayoutTensor[type, layout_unknown_3d](
            output_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](4, 2, 4)),
        ),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(4):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg6
# CHECK: 1.0 ,2.0 ,1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,3.0 ,4.0 ,
fn test_tile_eg6() raises:
    print("== test_tile_eg6")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2](uninitialized=True)

    var input = LayoutTensor[type, Layout.row_major(2, 2)](input_stack)

    input[0, 0] = 1
    input[0, 1] = 2
    input[1, 0] = 3
    input[1, 1] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 1
    repeats[1] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 2 * 4](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(2, 4)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 4)),
        ),
    )

    print()
    for i in range(2):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg7
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
fn test_tile_eg7() raises:
    print("== test_tile_eg7")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2)](input_stack)

    input[0, 0] = 1
    input[0, 1] = 2
    input[1, 0] = 3
    input[1, 1] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 2](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 2)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](4, 2)),
        ),
    )

    print()
    for i in range(4):
        for j in range(2):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg8
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
fn test_tile_eg8() raises:
    print("== test_tile_eg8")
    alias rank = 2
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 4](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(1, 4)](input_stack)

    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[0, 3] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 2](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(2)](repeats_stack)

    repeats[0] = 4
    repeats[1] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 4](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 4)](output_stack)

    for i in range(4):
        for j in range(4):
            output[i, j] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_2d](
            input_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](1, 4)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](2)),
        ),
        LayoutTensor[type, layout_unknown_2d](
            output_stack,
            RuntimeLayout[layout_unknown_2d].row_major(IndexList[2](4, 4)),
        ),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg9
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
fn test_tile_eg9() raises:
    print("== test_tile_eg9")
    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2)](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 3](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(3)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 2
    repeats[2] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 4 * 2](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(4, 4, 2)](output_stack)

    for i in range(4):
        for j in range(4):
            for k in range(2):
                output[i, j, k] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_3d](
            input_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](2, 2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](3)),
        ),
        LayoutTensor[type, layout_unknown_3d](
            output_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](4, 4, 2)),
        ),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg10
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
fn test_tile_eg10() raises:
    print("== test_tile_eg10")
    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2)](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 3](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(3)](repeats_stack)

    repeats[0] = 3
    repeats[1] = 2
    repeats[2] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 6 * 4 * 6](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(6, 4, 6)](output_stack)

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_3d](
            input_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](2, 2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](3)),
        ),
        LayoutTensor[type, layout_unknown_3d](
            output_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](6, 4, 6)),
        ),
    )

    print()
    for i in range(6):
        for j in range(4):
            for k in range(6):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg11
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
fn test_tile_eg11() raises:
    print("== test_tile_eg11")
    alias rank = 3
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 3 * 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(3, 2, 2)](input_stack)

    input[0, 0, 0] = 0
    input[0, 0, 1] = 1
    input[0, 1, 0] = 2
    input[0, 1, 1] = 3

    input[1, 0, 0] = 4
    input[1, 0, 1] = 5
    input[1, 1, 0] = 6
    input[1, 1, 1] = 7

    input[2, 0, 0] = 8
    input[2, 0, 1] = 9
    input[2, 1, 0] = 10
    input[2, 1, 1] = 11

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 3](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(3)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 3
    repeats[2] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 6 * 6 * 2](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(6, 6, 2)](output_stack)

    for i in range(6):
        for j in range(6):
            for k in range(2):
                output[i, j, k] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_3d](
            input_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](3, 2, 2)),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](3)),
        ),
        LayoutTensor[type, layout_unknown_3d](
            output_stack,
            RuntimeLayout[layout_unknown_3d].row_major(IndexList[3](6, 6, 2)),
        ),
    )

    print()
    for i in range(6):
        for j in range(6):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg12
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg12() raises:
    print("== test_tile_eg12")
    alias rank = 4
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2](uninitialized=True)
    var input = LayoutTensor[type, Layout.row_major(1, 1, 2, 2)](input_stack)

    input[0, 0, 0, 0] = 0
    input[0, 0, 0, 1] = 1
    input[0, 0, 1, 0] = 2
    input[0, 0, 1, 1] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 4](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(4)](repeats_stack)

    repeats[0] = 1
    repeats[1] = 1
    repeats[2] = 2
    repeats[3] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 6](uninitialized=True)
    var output = LayoutTensor[type, Layout.row_major(1, 1, 4, 6)](output_stack)

    for i in range(1):
        for j in range(1):
            for k in range(4):
                for l in range(6):
                    output[i, j, k, l] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_4d](
            input_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](1, 1, 2, 2)
            ),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](4)),
        ),
        LayoutTensor[type, layout_unknown_4d](
            output_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](1, 1, 4, 6)
            ),
        ),
    )

    print()
    for i in range(1):
        for j in range(1):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABE: test_tile_eg13
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
fn test_tile_eg13() raises:
    print("== test_tile_eg13")
    alias rank = 4
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2 * 2](
        uninitialized=True
    )
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2, 2)](input_stack)

    input[0, 0, 0, 0] = 0
    input[0, 0, 0, 1] = 1
    input[0, 0, 1, 0] = 2
    input[0, 0, 1, 1] = 3

    input[0, 1, 0, 0] = 4
    input[0, 1, 0, 1] = 5
    input[0, 1, 1, 0] = 6
    input[0, 1, 1, 1] = 7

    input[1, 0, 0, 0] = 8
    input[1, 0, 0, 1] = 9
    input[1, 0, 1, 0] = 10
    input[1, 0, 1, 1] = 11

    input[1, 1, 0, 0] = 12
    input[1, 1, 0, 1] = 13
    input[1, 1, 1, 0] = 14
    input[1, 1, 1, 1] = 15

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 4](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(4)](repeats_stack)

    repeats[0] = 1
    repeats[1] = 2
    repeats[2] = 2
    repeats[3] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 2 * 4 * 4 * 6](
        uninitialized=True
    )
    var output = LayoutTensor[type, Layout.row_major(2, 4, 4, 6)](output_stack)

    for i in range(2):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    output[i, j, k, l] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_4d](
            input_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](2, 2, 2, 2)
            ),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](4)),
        ),
        LayoutTensor[type, layout_unknown_4d](
            output_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](2, 4, 4, 6)
            ),
        ),
    )

    print()
    for i in range(2):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABE: test_tile_eg14
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
fn test_tile_eg14() raises:
    print("== test_tile_eg14")
    alias rank = 4
    alias type = DType.float32

    var input_stack = InlineArray[Scalar[type], 2 * 2 * 2 * 2](
        uninitialized=True
    )
    var input = LayoutTensor[type, Layout.row_major(2, 2, 2, 2)](input_stack)

    input[0, 0, 0, 0] = 0
    input[0, 0, 0, 1] = 1
    input[0, 0, 1, 0] = 2
    input[0, 0, 1, 1] = 3

    input[0, 1, 0, 0] = 4
    input[0, 1, 0, 1] = 5
    input[0, 1, 1, 0] = 6
    input[0, 1, 1, 1] = 7

    input[1, 0, 0, 0] = 8
    input[1, 0, 0, 1] = 9
    input[1, 0, 1, 0] = 10
    input[1, 0, 1, 1] = 11

    input[1, 1, 0, 0] = 12
    input[1, 1, 0, 1] = 13
    input[1, 1, 1, 0] = 14
    input[1, 1, 1, 1] = 15

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats_stack = InlineArray[Scalar[type_repeats], 4](uninitialized=True)
    var repeats = LayoutTensor[type_repeats, Layout.row_major(4)](repeats_stack)

    repeats[0] = 2
    repeats[1] = 2
    repeats[2] = 2
    repeats[3] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output_stack = InlineArray[Scalar[type], 4 * 4 * 4 * 6](
        uninitialized=True
    )
    var output = LayoutTensor[type, Layout.row_major(4, 4, 4, 6)](output_stack)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    output[i, j, k, l] = 0

    tile[type, type_repeats](
        LayoutTensor[type, layout_unknown_4d](
            input_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](2, 2, 2, 2)
            ),
        ),
        LayoutTensor[type_repeats, layout_unknown_1d](
            repeats_stack,
            RuntimeLayout[layout_unknown_1d].row_major(IndexList[1](4)),
        ),
        LayoutTensor[type, layout_unknown_4d](
            output_stack,
            RuntimeLayout[layout_unknown_4d].row_major(
                IndexList[4](4, 4, 4, 6)
            ),
        ),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


fn main() raises:
    test_tile_eg1()
    test_tile_eg2()
    test_tile_eg3()
    test_tile_eg4()
    test_tile_eg5()
    test_tile_eg6()
    test_tile_eg7()
    test_tile_eg8()
    test_tile_eg9()
    test_tile_eg10()
    test_tile_eg11()
    test_tile_eg12()
    test_tile_eg13()
    test_tile_eg14()
