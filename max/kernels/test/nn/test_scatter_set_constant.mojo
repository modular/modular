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

from layout import Layout, LayoutTensor
from nn.gather_scatter import scatter_set_constant
from runtime.asyncrt import DeviceContextPtr


fn test_scatter_set_constant() raises:
    # TODO not sure why this doesn't work with InlineArray?
    var data_stack = InlineArray[Float32, 9](uninitialized=True)
    var data = LayoutTensor[DType.float32, Layout.row_major(3, 3)](
        data_stack
    ).fill(0.0)

    var indices = LayoutTensor[DType.int32, Layout.row_major(4, 2)](
        InlineArray[Int32, 4 * 2](uninitialized=True),
    )

    indices[0, 0] = 0
    indices[0, 1] = 1
    indices[1, 0] = 1
    indices[1, 1] = 2
    indices[2, 0] = 1
    indices[2, 1] = 3
    indices[3, 0] = 2
    indices[3, 1] = 0

    var fill_value: Float32 = 5.0
    var expected_output_stack = InlineArray[Float32, 3 * 3](uninitialized=True)
    var expected_output = LayoutTensor[DType.float32, Layout.row_major(3, 3)](
        expected_output_stack
    ).fill(0.0)

    expected_output[0, 1] = 5.0
    expected_output[1, 2] = 5.0
    expected_output[1, 3] = 5.0
    expected_output[2, 0] = 5.0

    var ctx = DeviceContextPtr()

    scatter_set_constant[target="cpu",](data, indices, fill_value, ctx)

    for i in range(3):
        for j in range(3):
            if data[i, j] != expected_output[i, j]:
                raise Error(
                    "data[",
                    i,
                    ", ",
                    j,
                    "] = ",
                    data[i, j],
                    " != ",
                    expected_output[i, j],
                )


def main():
    test_scatter_set_constant()
