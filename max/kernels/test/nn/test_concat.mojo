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


from layout import (
    Coord,
    TensorLayout,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext
from nn.concat import _concat_parallel, _concat_serial, concat

from std.utils import StaticTuple


def _tuple_to_list[
    LayoutType: TensorLayout,
    origin: ImmOrigin,
    //,
    dtype: DType,
](
    elems: StaticTuple[
        TileTensor[dtype, LayoutType, origin],
        ...,
    ]
) -> List[
    TileTensor[dtype, LayoutType, origin]
]:
    var output = List[TileTensor[dtype, LayoutType, origin]](
        capacity=len(elems)
    )
    for i in range(len(elems)):
        output.append(elems[i])
    return output^


def test_concat() raises:
    print("== test_concat")

    comptime dtype = DType.float32
    comptime rank = 4
    comptime concat_axis = 2

    comptime l1 = row_major[2, 2, 1, 2]()
    comptime l2 = row_major[2, 2, 2, 2]()
    comptime l3 = row_major[2, 2, 3, 2]()
    # One backing buffer, because concat takes all inputs at one origin.
    comptime in_len = l1.product() + l2.product() + l3.product()
    var x_stack = Array[Scalar[dtype], in_len](fill=0)
    for i in range(l1.product(), l1.product() + l2.product()):
        x_stack[i] = 1
    for i in range(l1.product() + l2.product(), in_len):
        x_stack[i] = 2
    var x_span = Span(x_stack)
    var x1 = TileTensor(x_span[: l1.product()], l1)
    var x2 = TileTensor(x_span[l1.product() : l1.product() + l2.product()], l2)
    var x3 = TileTensor(x_span[l1.product() + l2.product() :], l3)

    comptime out_layout = row_major[2, 2, 6, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)
    var x1_dyn = x1.make_dynamic[.int64]()

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmOrigin(origin_of(x_stack))],
        3,
    ](
        x1_dyn.as_imm(),
        x2.make_dynamic[.int64]().as_imm(),
        x3.make_dynamic[.int64]().as_imm(),
    )

    @inline(.always)
    def epilogue_plus_one[
        c_type: DType, width: SIMDLength, *, alignment: Int
    ](coord: Coord, val: SIMD[c_type, width]) {var output}:
        comptime assert output.flat_rank >= coord.flat_rank
        output.store(
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    # The epilogue must capture `output` to store into it, and concat also
    # takes `output` mutably, so the output arg has to erase its origin for
    # the call to pass the origin checker.
    concat[dtype](
        output.make_dynamic[.int64]().as_unsafe_any_origin(),
        concat_axis,
        input_tuple,
        epilogue_plus_one,
        DeviceContext.cpu(),
    )

    # CHECK: == test_concat
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    var output_flat = output.reshape(row_major(output.num_elements()))
    for i in range(output.layout.product()):
        print(output_flat[i])


def test_concat_parallel() raises:
    print("== test_concat_parallel")

    comptime dtype = DType.float32
    comptime rank = 4
    comptime concat_axis = 2

    comptime l1 = row_major[2, 2, 1, 2]()
    comptime l2 = row_major[2, 2, 2, 2]()
    comptime l3 = row_major[2, 2, 3, 2]()
    # One backing buffer, because concat takes all inputs at one origin.
    comptime in_len = l1.product() + l2.product() + l3.product()
    var x_stack = Array[Scalar[dtype], in_len](fill=0)
    for i in range(l1.product(), l1.product() + l2.product()):
        x_stack[i] = 1
    for i in range(l1.product() + l2.product(), in_len):
        x_stack[i] = 2
    var x_span = Span(x_stack)
    var x1 = TileTensor(x_span[: l1.product()], l1)
    var x2 = TileTensor(x_span[l1.product() : l1.product() + l2.product()], l2)
    var x3 = TileTensor(x_span[l1.product() + l2.product() :], l3)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()
    var x3_dyn = x3.make_dynamic[.int64]()

    comptime out_layout = row_major[2, 2, 6, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmOrigin(origin_of(x_stack))],
        3,
    ](
        x1_dyn.as_imm(),
        x2_dyn.as_imm(),
        x3_dyn.as_imm(),
    )

    @inline(.always)
    def epilogue_plus_one[
        c_type: DType, width: SIMDLength, *, alignment: Int
    ](coord: Coord, val: SIMD[c_type, width]) {var output}:
        comptime assert output.flat_rank >= coord.flat_rank
        output.store(
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    var input_vec = _tuple_to_list(input_tuple)
    # The epilogue must capture `output` to store into it, and the kernel
    # also takes `output` mutably, so the output arg has to erase its
    # origin for the call to pass the origin checker.
    _concat_parallel[dtype, has_epilogue=True](
        output.make_dynamic[.int64]().as_unsafe_any_origin(),
        concat_axis,
        input_vec,
        epilogue_plus_one,
    )

    # CHECK: == test_concat_parallel
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    # CHECK-COUNT-2: 1.0
    # CHECK-COUNT-4: 2.0
    # CHECK-COUNT-6: 3.0
    var output_flat = output.reshape(row_major(output.num_elements()))
    for i in range(output.layout.product()):
        print(output_flat[i])


# CHECK-LABEL: test_concat_inner
def test_concat_inner() raises:
    print("== test_concat_inner")

    comptime dtype = DType.float32
    comptime rank = 5
    comptime concat_axis = 2

    comptime l1 = row_major[1, 1, 1, 2, 2]()
    comptime l2 = row_major[1, 1, 2, 2, 2]()
    comptime l3 = row_major[1, 1, 3, 2, 2]()
    # One backing buffer, because concat takes all inputs at one origin.
    comptime in_len = l1.product() + l2.product() + l3.product()
    var x_stack = Array[Scalar[dtype], in_len](fill=0)
    for i in range(l1.product(), l1.product() + l2.product()):
        x_stack[i] = 1
    for i in range(l1.product() + l2.product(), in_len):
        x_stack[i] = 2
    var x_span = Span(x_stack)
    var x1 = TileTensor(x_span[: l1.product()], l1)
    var x2 = TileTensor(x_span[l1.product() : l1.product() + l2.product()], l2)
    var x3 = TileTensor(x_span[l1.product() + l2.product() :], l3)

    var x1_dyn = x1.make_dynamic[.int64]()
    var x2_dyn = x2.make_dynamic[.int64]()
    var x3_dyn = x3.make_dynamic[.int64]()

    comptime out_layout = row_major[1, 1, 6, 2, 2]()
    var out_stack = Array[Scalar[dtype], out_layout.product()](fill=-1)
    var output = TileTensor(out_stack, out_layout)

    var input_tuple = StaticTuple[
        TileTensor[dtype, x1_dyn.LayoutType, ImmOrigin(origin_of(x_stack))],
        3,
    ](
        x1_dyn.as_imm(),
        x2_dyn.as_imm(),
        x3_dyn.as_imm(),
    )

    var input_vec = _tuple_to_list(input_tuple)

    @inline(.always)
    def epilogue_plus_one[
        c_type: DType, width: SIMDLength, *, alignment: Int
    ](coord: Coord, val: SIMD[c_type, width]) {var output}:
        comptime assert output.flat_rank >= coord.flat_rank
        output.store(
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    # As in test_concat_parallel: the epilogue captures `output`, so the
    # output arg has to erase its origin.
    _concat_serial[dtype, has_epilogue=True](
        output.make_dynamic[.int64]().as_unsafe_any_origin(),
        concat_axis,
        input_vec,
        epilogue_plus_one,
    )

    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-8: 2.0
    # CHECK-COUNT-12: 3.0
    var output_flat = output.reshape(row_major(output.num_elements()))
    for i in range(output.layout.product()):
        print(output_flat[i])


def main() raises:
    test_concat()
    test_concat_parallel()
    test_concat_inner()
