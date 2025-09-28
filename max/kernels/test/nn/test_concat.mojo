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


from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from nn.concat import _concat_parallel, _concat_serial, concat

from utils import Index, IndexList, StaticTuple


fn _tuple_to_list[
    elems_layout: Layout, //,
    dtype: DType,
](
    elems: StaticTuple[LayoutTensor[dtype, elems_layout, MutableAnyOrigin], *_]
) -> List[LayoutTensor[dtype, elems_layout, MutableAnyOrigin]]:
    var output = List[LayoutTensor[dtype, elems_layout, MutableAnyOrigin]](
        capacity=len(elems)
    )
    for i in range(len(elems)):
        output.append(elems[i])
    return output^


def test_concat():
    print("== test_concat")

    alias dtype = DType.float32
    alias rank = 4
    alias concat_axis = 2

    alias l1 = Layout.row_major(2, 2, 1, 2)
    alias l2 = Layout.row_major(2, 2, 2, 2)
    alias l3 = Layout.row_major(2, 2, 3, 2)
    alias s1 = IndexList[rank](2, 2, 1, 2)
    alias s2 = IndexList[rank](2, 2, 2, 2)
    alias s3 = IndexList[rank](2, 2, 3, 2)

    alias layout = Layout.row_major[rank]()

    var x1_stack = InlineArray[Scalar[dtype], l1.size()](uninitialized=True)
    var x2_stack = InlineArray[Scalar[dtype], l2.size()](uninitialized=True)
    var x3_stack = InlineArray[Scalar[dtype], l3.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, l1](x1_stack).fill(0)
    var x2 = LayoutTensor[dtype, l2](x2_stack).fill(1)
    var x3 = LayoutTensor[dtype, l3](x3_stack).fill(2)
    var x1_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x1.ptr, RuntimeLayout[layout].row_major(s1)
    )
    var x2_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x2.ptr, RuntimeLayout[layout].row_major(s2)
    )
    var x3_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x3.ptr, RuntimeLayout[layout].row_major(s3)
    )

    alias out_layout = Layout.row_major(2, 2, 6, 2)
    alias out_shape = IndexList[rank](2, 2, 6, 2)
    var out_stack = InlineArray[Scalar[dtype], out_layout.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_layout](out_stack).fill(-1)
    var output_dyn = LayoutTensor[dtype, layout](
        output.ptr, RuntimeLayout[layout].row_major(out_shape)
    )

    var input_tuple = StaticTuple[
        LayoutTensor[dtype, layout, MutableAnyOrigin], 3
    ](x1_dyn, x2_dyn, x3_dyn)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[dtype, width]](val + 1),
        )

    concat[dtype, False, epilogue_fn=epilogue_plus_one](
        output_dyn, concat_axis, input_tuple
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
    var output_flat = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(output.size())
        ),
    )
    for i in range(out_layout.size()):
        print(output_flat.load[1](Index(i)))


def test_concat_parallel():
    print("== test_concat_parallel")

    alias dtype = DType.float32
    alias rank = 4
    alias concat_axis = 2

    alias l1 = Layout.row_major(2, 2, 1, 2)
    alias l2 = Layout.row_major(2, 2, 2, 2)
    alias l3 = Layout.row_major(2, 2, 3, 2)
    alias s1 = IndexList[rank](2, 2, 1, 2)
    alias s2 = IndexList[rank](2, 2, 2, 2)
    alias s3 = IndexList[rank](2, 2, 3, 2)

    var x1_stack = InlineArray[Scalar[dtype], l1.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, l1](x1_stack).fill(0)
    var x2_stack = InlineArray[Scalar[dtype], l2.size()](uninitialized=True)
    var x2 = LayoutTensor[dtype, l2](x2_stack).fill(1)
    var x3_stack = InlineArray[Scalar[dtype], l3.size()](uninitialized=True)
    var x3 = LayoutTensor[dtype, l3](x3_stack).fill(2)
    alias layout = Layout.row_major[rank]()
    var x1_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x1.ptr, RuntimeLayout[layout].row_major(s1)
    )
    var x2_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x2.ptr, RuntimeLayout[layout].row_major(s2)
    )
    var x3_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x3.ptr, RuntimeLayout[layout].row_major(s3)
    )

    alias out_layout = Layout.row_major(2, 2, 6, 2)
    alias out_shape = IndexList[rank](2, 2, 6, 2)
    var out_stack = InlineArray[Scalar[dtype], out_layout.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_layout](out_stack).fill(-1)
    var output_dyn = LayoutTensor[dtype, layout](
        output.ptr, RuntimeLayout[layout].row_major(out_shape)
    )

    var input_tuple = StaticTuple[
        LayoutTensor[dtype, layout, MutableAnyOrigin], 3
    ](x1_dyn, x2_dyn, x3_dyn)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[dtype, width]](val + 1),
        )

    var input_vec = _tuple_to_list(input_tuple)
    _concat_parallel[dtype, epilogue_plus_one](
        output_dyn, concat_axis, input_vec
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
    var output_flat = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(output.size())
        ),
    )
    for i in range(out_layout.size()):
        print(output_flat.load[1](Index(i)))


# CHECK-LABEL: test_concat_inner
def test_concat_inner():
    print("== test_concat_inner")

    alias dtype = DType.float32
    alias rank = 5
    alias concat_axis = 2

    alias l1 = Layout.row_major(1, 1, 1, 2, 2)
    alias l2 = Layout.row_major(1, 1, 2, 2, 2)
    alias l3 = Layout.row_major(1, 1, 3, 2, 2)
    alias s1 = IndexList[rank](1, 1, 1, 2, 2)
    alias s2 = IndexList[rank](1, 1, 2, 2, 2)
    alias s3 = IndexList[rank](1, 1, 3, 2, 2)

    var x1_stack = InlineArray[Scalar[dtype], l1.size()](uninitialized=True)
    var x2_stack = InlineArray[Scalar[dtype], l2.size()](uninitialized=True)
    var x3_stack = InlineArray[Scalar[dtype], l3.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, l1](x1_stack).fill(0)
    var x2 = LayoutTensor[dtype, l2](x2_stack).fill(1)
    var x3 = LayoutTensor[dtype, l3](x3_stack).fill(2)
    alias layout = Layout.row_major[rank]()
    var x1_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x1.ptr, RuntimeLayout[layout].row_major(s1)
    )
    var x2_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x2.ptr, RuntimeLayout[layout].row_major(s2)
    )
    var x3_dyn = LayoutTensor[dtype, layout, MutableAnyOrigin](
        x3.ptr, RuntimeLayout[layout].row_major(s3)
    )

    alias out_shape = IndexList[rank](1, 1, 6, 2, 2)
    alias out_layout = Layout.row_major(1, 1, 6, 2, 2)
    var out_stack = InlineArray[Scalar[dtype], out_layout.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_layout](out_stack).fill(-1)
    var output_dyn = LayoutTensor[dtype, layout](
        output.ptr, RuntimeLayout[layout].row_major(out_shape)
    )

    var input_list = StaticTuple[
        LayoutTensor[dtype, layout, MutableAnyOrigin], 3
    ](x1_dyn, x2_dyn, x3_dyn)

    var input_vec = _tuple_to_list(input_list)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[dtype, width]](val + 1),
        )

    _concat_serial[dtype, epilogue_plus_one](output_dyn, concat_axis, input_vec)

    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-8: 2.0
    # CHECK-COUNT-12: 3.0
    var output_flat = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(output.size())
        ),
    )
    for i in range(out_layout.size()):
        print(output_flat.load[1](Index(i)))


def main():
    test_concat()
    test_concat_parallel()
    test_concat_inner()
