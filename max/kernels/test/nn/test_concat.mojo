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

from collections import OptionalReg

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.concat import (
    _concat_parallel,
    _concat_serial,
    concat,
    elementwise_epilogue_type,
)

from utils import IndexList, StaticTuple


fn _tuple_to_list[
    type: DType,
    rank: Int,
](elems: StaticTuple[NDBuffer[type, rank, MutableAnyOrigin], *_]) -> List[
    NDBuffer[type, rank, MutableAnyOrigin]
]:
    var output = List[NDBuffer[type, rank, MutableAnyOrigin]](
        capacity=len(elems)
    )
    for i in range(len(elems)):
        output.append(elems[i])
    return output^


def test_concat():
    print("== test_concat")

    alias type = DType.float32
    alias rank = 4
    alias concat_axis = 2

    alias s1 = DimList(2, 2, 1, 2, 0)
    alias s2 = DimList(2, 2, 2, 2, 0)
    alias s3 = DimList(2, 2, 3, 2, 0)

    var x1_stack = InlineArray[Scalar[type], Int(s1.product())](
        uninitialized=True
    )
    var x1 = NDBuffer[type, rank, _, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[type], Int(s2.product())](
        uninitialized=True
    )
    var x2 = NDBuffer[type, rank, _, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[type], Int(s3.product())](
        uninitialized=True
    )
    var x3 = NDBuffer[type, rank, _, s3](x3_stack)
    x1.fill(0)
    x2.fill(1)
    x3.fill(2)
    var x1_dyn = NDBuffer[type, rank](x1.data, s1)
    var x2_dyn = NDBuffer[type, rank](x2.data, s2)
    var x3_dyn = NDBuffer[type, rank](x3.data, s3)

    alias out_shape = DimList(2, 2, 6, 2, 0)
    var out_stack = InlineArray[Scalar[type], Int(out_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, rank, _, out_shape](out_stack)
    output.fill(-1)
    var output_dyn = NDBuffer[type, rank](output.data, out_shape)

    var input_tuple = StaticTuple[NDBuffer[type, rank, MutableAnyOrigin], 3](
        x1_dyn, x2_dyn, x3_dyn
    )

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[type, width]](val + 1),
        )

    concat[rank, type, False, epilogue_fn=epilogue_plus_one](
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
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


def test_concat_parallel():
    print("== test_concat_parallel")

    alias type = DType.float32
    alias rank = 4
    alias concat_axis = 2

    alias s1 = DimList(2, 2, 1, 2, 0)
    alias s2 = DimList(2, 2, 2, 2, 0)
    alias s3 = DimList(2, 2, 3, 2, 0)

    var x1_stack = InlineArray[Scalar[type], Int(s1.product())](
        uninitialized=True
    )
    var x1 = NDBuffer[type, rank, _, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[type], Int(s2.product())](
        uninitialized=True
    )
    var x2 = NDBuffer[type, rank, _, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[type], Int(s3.product())](
        uninitialized=True
    )
    var x3 = NDBuffer[type, rank, _, s3](x3_stack)
    x1.fill(0)
    x2.fill(1)
    x3.fill(2)
    var x1_dyn = NDBuffer[type, rank](x1.data, s1)
    var x2_dyn = NDBuffer[type, rank](x2.data, s2)
    var x3_dyn = NDBuffer[type, rank](x3.data, s3)

    alias out_shape = DimList(2, 2, 6, 2, 0)
    var out_stack = InlineArray[Scalar[type], Int(out_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, rank, _, out_shape](out_stack)
    output.fill(-1)
    var output_dyn = NDBuffer[type, rank](output.data, out_shape)

    var input_tuple = StaticTuple[NDBuffer[type, rank, MutableAnyOrigin], 3](
        x1_dyn, x2_dyn, x3_dyn
    )

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[type, width]](val + 1),
        )

    var input_vec = _tuple_to_list(input_tuple)
    _concat_parallel[rank, type, epilogue_plus_one](
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
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


# CHECK-LABEL: test_concat_inner
def test_concat_inner():
    print("== test_concat_inner")

    alias type = DType.float32
    alias rank = 5
    alias concat_axis = 2

    alias s1 = DimList(1, 1, 1, 2, 2)
    alias s2 = DimList(1, 1, 2, 2, 2)
    alias s3 = DimList(1, 1, 3, 2, 2)

    var x1_stack = InlineArray[Scalar[type], Int(s1.product())](
        uninitialized=True
    )
    var x1 = NDBuffer[type, rank, _, s1](x1_stack)
    var x2_stack = InlineArray[Scalar[type], Int(s2.product())](
        uninitialized=True
    )
    var x2 = NDBuffer[type, rank, _, s2](x2_stack)
    var x3_stack = InlineArray[Scalar[type], Int(s3.product())](
        uninitialized=True
    )
    var x3 = NDBuffer[type, rank, _, s3](x3_stack)
    x1.fill(0)
    x2.fill(1)
    x3.fill(2)
    var x1_dyn = NDBuffer[type, rank](x1.data, s1)
    var x2_dyn = NDBuffer[type, rank](x2.data, s2)
    var x3_dyn = NDBuffer[type, rank](x3.data, s3)

    alias out_shape = DimList(1, 1, 6, 2, 2)
    var out_stack = InlineArray[Scalar[type], Int(out_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, rank, _, out_shape](out_stack)
    output.fill(-1)
    var output_dyn = NDBuffer[type, rank](output.data, out_shape)

    var input_list = StaticTuple[NDBuffer[type, rank, MutableAnyOrigin], 3](
        x1_dyn, x2_dyn, x3_dyn
    )

    var input_vec = _tuple_to_list(input_list)

    @parameter
    @always_inline
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[type, width]](val + 1),
        )

    _concat_serial[rank, type, epilogue_plus_one](
        output_dyn, concat_axis, input_vec
    )

    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-8: 2.0
    # CHECK-COUNT-12: 3.0
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


def main():
    test_concat()
    test_concat_parallel()
    test_concat_inner()
