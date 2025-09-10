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


from layout import (
    LayoutTensor,
    Layout,
    IntTuple,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.concat import (
    _concat_parallel,
    _concat_serial,
    concat,
)

from utils import IndexList, StaticTuple


fn _tuple_to_list[
    origin: ImmutableOrigin, //,
    dtype: DType,
    rank: Int,
](
    elems: StaticTuple[
        LayoutTensor[dtype, Layout.row_major[rank](), origin], *_
    ]
) -> List[LayoutTensor[dtype, Layout.row_major[rank](), origin]]:
    var output = List[LayoutTensor[dtype, Layout.row_major[rank](), origin]](
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

    alias s1 = Layout.row_major(2, 2, 1, 2)
    alias s2 = Layout.row_major(2, 2, 2, 2)
    alias s3 = Layout.row_major(2, 2, 3, 2)

    var x1_stack = InlineArray[Scalar[dtype], s1.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[dtype], s2.size()](uninitialized=True)
    var x2 = LayoutTensor[dtype, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[dtype], s3.size()](uninitialized=True)
    var x3 = LayoutTensor[dtype, s3](x3_stack)
    _ = x1.fill(0)
    _ = x2.fill(1)
    _ = x3.fill(2)
    alias out_shape = Layout.row_major(2, 2, 6, 2)
    var out_stack = InlineArray[Scalar[dtype], out_shape.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_shape](out_stack)
    _ = output.fill(-1)
    var output_dyn = LayoutTensor[dtype, Layout.row_major[rank]()](
        output.ptr,
        RuntimeLayout[Layout.row_major[rank]()].row_major(
            output.runtime_layout.shape.value.canonicalize()
        ),
    )
    alias lyt = Layout.row_major[output_dyn.rank]()
    alias dyn_origin = ImmutableOrigin.cast_from[
        __origin_of(x1_stack, x2_stack, x3_stack)
    ]
    var x1_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x1.ptr,
        RuntimeLayout[lyt].row_major(
            x1.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x2_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x2.ptr,
        RuntimeLayout[lyt].row_major(
            x2.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x3_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x3.ptr,
        RuntimeLayout[lyt].row_major(
            x3.runtime_layout.shape.value.canonicalize()
        ),
    )

    var input_tuple = StaticTuple[
        LayoutTensor[dtype, Layout.row_major[output_dyn.rank](), dyn_origin],
        3,
    ](x1_dyn, x2_dyn, x3_dyn)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](coords: IndexList[_rank], val: SIMD[c_type, width]):
        var idx = output.runtime_layout(
            RuntimeTuple[fill_like(output.layout.shape, UNKNOWN_VALUE)](coords)
        )
        output.ptr.store[width=width](
            idx,
            rebind[SIMD[dtype, width]](val + 1),
        )

    concat[
        input_origin=dyn_origin, dtype, False, epilogue_fn=epilogue_plus_one
    ](output_dyn, concat_axis, input_tuple)

    var flattened = LayoutTensor[output.dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](output.size())
        ),
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
    for i in range(out_shape.size()):
        var idx = flattened.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
        )
        print(flattened.ptr[idx])


def test_concat_parallel():
    print("== test_concat_parallel")

    alias dtype = DType.float32
    alias rank = 4
    alias concat_axis = 2

    alias s1 = Layout.row_major(2, 2, 1, 2)
    alias s2 = Layout.row_major(2, 2, 2, 2)
    alias s3 = Layout.row_major(2, 2, 3, 2)

    var x1_stack = InlineArray[Scalar[dtype], s1.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[dtype], s2.size()](uninitialized=True)
    var x2 = LayoutTensor[dtype, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[dtype], s3.size()](uninitialized=True)
    var x3 = LayoutTensor[dtype, s3](x3_stack)
    _ = x1.fill(0)
    _ = x2.fill(1)
    _ = x3.fill(2)
    alias lyt = Layout.row_major[rank]()
    alias dyn_origin = ImmutableOrigin.cast_from[
        __origin_of(x1_stack, x2_stack, x3_stack)
    ]
    var x1_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x1.ptr,
        RuntimeLayout[lyt].row_major(
            x1.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x2_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x2.ptr,
        RuntimeLayout[lyt].row_major(
            x2.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x3_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x3.ptr,
        RuntimeLayout[lyt].row_major(
            x3.runtime_layout.shape.value.canonicalize()
        ),
    )

    alias out_shape = Layout.row_major(2, 2, 6, 2)
    var out_stack = InlineArray[Scalar[dtype], out_shape.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_shape](out_stack)
    _ = output.fill(-1)
    var output_dyn = LayoutTensor[dtype, lyt](
        output.ptr,
        RuntimeLayout[lyt].row_major(
            output.runtime_layout.shape.value.canonicalize()
        ),
    )

    var input_tuple = StaticTuple[LayoutTensor[dtype, lyt, dyn_origin], 3](
        x1_dyn, x2_dyn, x3_dyn
    )

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](coords: IndexList[_rank], val: SIMD[c_type, width]):
        var idx = output.runtime_layout(
            RuntimeTuple[fill_like(output.layout.shape, UNKNOWN_VALUE)](coords)
        )
        output.ptr.store[width=width](
            idx,
            rebind[SIMD[dtype, width]](val + 1),
        )

    var input_vec = _tuple_to_list(input_tuple)
    _concat_parallel[dtype, epilogue_plus_one](
        output_dyn, concat_axis, input_vec
    )

    var flattened = LayoutTensor[output.dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](output.size())
        ),
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
    for i in range(output.size()):
        var idx = flattened.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
        )
        print(flattened.ptr[idx])


# CHECK-LABEL: test_concat_inner
def test_concat_inner():
    print("== test_concat_inner")

    alias dtype = DType.float32
    alias rank = 5
    alias concat_axis = 2

    alias s1 = Layout.row_major(1, 1, 1, 2, 2)
    alias s2 = Layout.row_major(1, 1, 2, 2, 2)
    alias s3 = Layout.row_major(1, 1, 3, 2, 2)

    var x1_stack = InlineArray[Scalar[dtype], s1.size()](uninitialized=True)
    var x1 = LayoutTensor[dtype, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[dtype], s2.size()](uninitialized=True)
    var x2 = LayoutTensor[dtype, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[dtype], s3.size()](uninitialized=True)
    var x3 = LayoutTensor[dtype, s3](x3_stack)
    _ = x1.fill(0)
    _ = x2.fill(1)
    _ = x3.fill(2)
    alias lyt = Layout.row_major[rank]()
    alias dyn_origin = ImmutableOrigin.cast_from[
        __origin_of(x1_stack, x2_stack, x3_stack)
    ]
    var x1_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x1.ptr,
        RuntimeLayout[lyt].row_major(
            x1.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x2_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x2.ptr,
        RuntimeLayout[lyt].row_major(
            x2.runtime_layout.shape.value.canonicalize()
        ),
    )
    var x3_dyn = LayoutTensor[dtype, lyt, dyn_origin](
        x3.ptr,
        RuntimeLayout[lyt].row_major(
            x3.runtime_layout.shape.value.canonicalize()
        ),
    )

    alias out_shape = Layout.row_major(1, 1, 6, 2, 2)
    var out_stack = InlineArray[Scalar[dtype], out_shape.size()](
        uninitialized=True
    )
    var output = LayoutTensor[dtype, out_shape](out_stack)
    _ = output.fill(-1)
    var output_dyn = LayoutTensor[dtype, lyt](
        output.ptr,
        RuntimeLayout[lyt].row_major(
            output.runtime_layout.shape.value.canonicalize()
        ),
    )

    var input_list = StaticTuple[LayoutTensor[dtype, lyt, dyn_origin], 3](
        x1_dyn, x2_dyn, x3_dyn
    )

    var input_vec = _tuple_to_list(input_list)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](coords: IndexList[_rank], val: SIMD[c_type, width]):
        var idx = output.runtime_layout(
            RuntimeTuple[fill_like(output.layout.shape, UNKNOWN_VALUE)](coords)
        )
        output.ptr.store[width=width](
            idx,
            rebind[SIMD[dtype, width]](val + 1),
        )

    _concat_serial[dtype, epilogue_plus_one](output_dyn, concat_axis, input_vec)

    var flattened = LayoutTensor[output.dtype, Layout.row_major(UNKNOWN_VALUE)](
        output.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](output.size())
        ),
    )
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-8: 2.0
    # CHECK-COUNT-12: 3.0
    for i in range(out_shape.size()):
        var idx = flattened.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
        )
        print(flattened.ptr[idx])


def main():
    test_concat()
    test_concat_parallel()
    test_concat_inner()
