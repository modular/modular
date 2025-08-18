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

from builtin.variadics import VariadicOf
from sys import alignof
from ._mixed_layout import MixedLayout
from ._mixed_tuple import MixedIntTuple, MixedIntTupleLike, ComptimeInt, Idx


struct MixedLayoutTensor[
    dtype: DType,
    shape_types: VariadicOf[MixedIntTupleLike],
    stride_types: VariadicOf[MixedIntTupleLike], //,
    alignment: Int = alignof[dtype](),
]:
    var ptr: UnsafePointer[Scalar[dtype]]

    var layout: MixedLayout[
        shape_types=shape_types,
        stride_types=stride_types,
    ]

    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[dtype]],
        layout: MixedLayout[shape_types, stride_types],
    ):
        self.ptr = ptr
        self.layout = layout

    fn __getitem__[
        index_type: MixedIntTupleLike
    ](self, arg: index_type) -> SIMD[dtype, 1]:
        return self.ptr[self.layout(arg)]

    fn __setitem__[
        index_type: MixedIntTupleLike
    ](self, arg: index_type, value: SIMD[dtype, 1]):
        self.ptr[self.layout(arg)] = value


fn distribute[
    thread_shape_0: Int,
    thread_shape_1: Int,
    thread_stride_0: Int,
    thread_stride_1: Int,
    data_shape_0: Int,
    data_shape_1: Int,
    data_stride_0: Int,
    data_stride_1: Int, //,
    dtype: DType,
    thread_layout: MixedLayout[
        MixedIntTuple[
            ComptimeInt[thread_shape_0], ComptimeInt[thread_shape_1]
        ]._get_variadic_pack(),
        MixedIntTuple[
            ComptimeInt[thread_stride_0], ComptimeInt[thread_stride_1]
        ]._get_variadic_pack(),
    ],
](
    data_layout_tensor: MixedLayoutTensor[
        dtype=dtype,
        shape_types = MixedIntTuple[
            ComptimeInt[data_shape_0], ComptimeInt[data_shape_1]
        ]._get_variadic_pack(),
        stride_types = MixedIntTuple[
            ComptimeInt[data_stride_0], ComptimeInt[data_stride_1]
        ]._get_variadic_pack(),
    ],
    thread_id: Int,
) -> MixedLayoutTensor[
    dtype = data_layout_tensor.dtype,
    shape_types = MixedIntTuple[
        ComptimeInt[data_shape_0 // thread_shape_0],
        ComptimeInt[data_shape_1 // thread_shape_1],
    ]._get_variadic_pack(),
    stride_types = MixedIntTuple[
        ComptimeInt[data_stride_0 * thread_shape_0],
        ComptimeInt[data_stride_1 * thread_shape_1],
    ]._get_variadic_pack(),
]:
    """A simplified implementation of LayoutTensor.distribute on MixedLayoutTensor.
    """

    var offset: UInt = 0

    @parameter
    for i in range(len(thread_layout.stride)):
        alias stride_i = Int(thread_layout.stride[i].value())
        alias shape_i = Int(thread_layout.shape[i].value())
        var thread_coord_i = (thread_id // stride_i) % shape_i
        offset += thread_coord_i * Int(
            data_layout_tensor.layout.stride[i].value()
        )

    alias shape = MixedIntTuple(
        ComptimeInt[data_shape_0 // thread_shape_0](),
        ComptimeInt[data_shape_1 // thread_shape_1](),
    )

    alias stride = MixedIntTuple(
        ComptimeInt[data_stride_0 * thread_shape_0](),
        ComptimeInt[data_stride_1 * thread_shape_1](),
    )

    var frag_layout = MixedLayout(
        shape=shape,
        stride=stride,
    )

    return MixedLayoutTensor[dtype = data_layout_tensor.dtype,](
        UnsafePointer(to=data_layout_tensor.ptr[offset]),
        rebind[
            MixedLayout[
                shape_types = __type_of(shape)._get_variadic_pack(),
                stride_types = __type_of(stride)._get_variadic_pack(),
            ]
        ](frag_layout),
    )
