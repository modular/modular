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

from sys.info import simd_width_of

from layout._fillers import arange
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.softmax import logsoftmax, softmax_2_pass

from testing import assert_equal

from utils import IndexList


# CHECK-LABEL: test_logsoftmax
fn test_logsoftmax() raises:
    print("== test_logsoftmax")
    alias type = DType.float32
    alias simd_width = simd_width_of[type]()

    fn logsoftmax_test_nd[rank: Int, shape: IndexList[rank]]() raises:
        alias layout = Layout.row_major(shape)
        var in_stack = InlineArray[Scalar[type], layout.size()](
            uninitialized=True
        )
        var in_buf = LayoutTensor[type, layout](in_stack)
        var out_stack = InlineArray[Scalar[type], layout.size()](
            uninitialized=True
        )
        arange(in_buf)
        var out_buf = LayoutTensor[type, layout](out_stack).fill(0)

        logsoftmax[type, simd_width, rank](in_buf, out_buf, rank - 1)

        var out_buf_flat = LayoutTensor[type, Layout.row_major(UNKNOWN_VALUE)](
            out_buf.ptr,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](out_buf.size())
            ),
        )
        for i in range(out_buf.size()):
            print(out_buf_flat.load[width=1](IndexList[1](i)))

    logsoftmax_test_nd[1, IndexList[1](5)]()

    # CHECK: -4.45191{{[0-9]+}}
    # CHECK-NEXT: -3.451914{{[0-9]+}}
    # CHECK-NEXT: -2.451914{{[0-9]+}}
    # CHECK-NEXT: -1.451914{{[0-9]+}}
    # CHECK-NEXT: -0.451914{{[0-9]+}}

    logsoftmax_test_nd[2, IndexList[2](3, 4)]()

    # CHECK: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}


# CHECK-LABEL: test_softmax_2pass
fn test_softmax_2pass():
    print("== test_softmax_2pass")
    alias type = DType.float32
    alias simd_width = simd_width_of[type]()
    alias sz = 5

    var in_stack = InlineArray[Scalar[type], sz](uninitialized=True)
    var in_buf = LayoutTensor[type, Layout.row_major(sz)](in_stack)
    for i in range(sz):
        in_buf[i] = i
    var out_stack = InlineArray[Scalar[type], sz](uninitialized=True)
    var out_buf = LayoutTensor[type, Layout.row_major(sz)](out_stack).fill(0)

    softmax_2_pass[simd_width, type](out_buf, in_buf)

    for i in range(sz):
        print(out_buf[i])

    # CHECK: 0.01165{{[0-9]+}}
    # CHECK-NEXT: 0.03168{{[0-9]+}}
    # CHECK-NEXT: 0.08612{{[0-9]+}}
    # CHECK-NEXT: 0.23412{{[0-9]+}}
    # CHECK-NEXT: 0.63640{{[0-9]+}}


def main():
    test_logsoftmax()
    test_softmax_2pass()
