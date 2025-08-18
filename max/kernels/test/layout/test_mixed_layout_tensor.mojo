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

from testing import assert_equal, assert_true
from layout._mixed_layout import MixedLayout, make_row_major
from layout._mixed_layout_tensor import MixedLayoutTensor, distribute
from layout._mixed_tuple import Idx, MixedIntTuple, ComptimeInt, RuntimeInt
from layout.int_tuple import IntTuple


fn main() raises:
    test_distribute()


fn test_distribute() raises:
    alias thread_layout = make_row_major(
        MixedIntTuple(ComptimeInt[2](), ComptimeInt[2]())
    )

    var ptr = InlineArray[Scalar[DType.uint32], 16](fill=-1).unsafe_ptr()

    alias data_layout_shape = MixedIntTuple[ComptimeInt[4], ComptimeInt[4]]
    alias data_layout_stride = MixedIntTuple[ComptimeInt[4], ComptimeInt[1]]
    alias data_layout_shape_types = data_layout_shape._get_variadic_pack()
    alias data_layout_stride_types = data_layout_stride._get_variadic_pack()

    var layout_tensor = MixedLayoutTensor[
        dtype = DType.uint32,
        shape_types=data_layout_shape_types,
        stride_types=data_layout_stride_types,
    ](
        ptr=ptr,
        layout=MixedLayout[
            shape_types=data_layout_shape_types,
            stride_types=data_layout_stride_types,
        ](
            shape=rebind[MixedIntTuple[*data_layout_shape_types]](
                data_layout_shape(ComptimeInt[4](), ComptimeInt[4]())
            ),
            stride=rebind[MixedIntTuple[*data_layout_stride_types]](
                data_layout_stride(ComptimeInt[4](), ComptimeInt[1]())
            ),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var frag = distribute[
            dtype = DType.uint32,
            thread_layout = rebind[
                MixedLayout[
                    shape_types = MixedIntTuple[
                        ComptimeInt[2], ComptimeInt[2]
                    ]._get_variadic_pack(),
                    stride_types = MixedIntTuple[
                        ComptimeInt[2], ComptimeInt[1]
                    ]._get_variadic_pack(),
                ]
            ](thread_layout),
        ](layout_tensor, th_id)

        # Fill the fragment positions with the thread id (0..3)
        for i in range(2):
            for j in range(2):
                frag[MixedIntTuple(Idx(i), Idx(j))] = counter
                counter += 1

    alias expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], expected[i])
