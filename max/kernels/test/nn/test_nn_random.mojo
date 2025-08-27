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

from random import seed

from layout import Layout, LayoutTensor
from nn.randn import random_normal


fn test_random_normal():
    seed(0)

    alias out_shape = Layout.row_major(2, 2)
    var output_stack = InlineArray[Float32, 4](uninitialized=True)
    var output = LayoutTensor[DType.float32, out_shape](output_stack).fill(0)

    random_normal[DType.float32, 0.0, 1.0](output)
    # CHECK-LABEL: == test_random_normal
    print("== test_random_normal")


fn main():
    test_random_normal()
