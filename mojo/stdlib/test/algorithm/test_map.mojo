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

from algorithm import map
from buffer import NDBuffer


# CHECK-LABEL: test_map
fn test_map():
    print("== test_map")

    var vector_stack = InlineArray[Float32, 5](1.0, 2.0, 3.0, 4.0, 5.0)
    var vector = NDBuffer[DType.float32, 1, _, 5](vector_stack)

    @parameter
    @__copy_capture(vector)
    fn add_two(idx: Int):
        vector[idx] = vector[idx] + 2

    map[add_two](len(vector))

    # CHECK: 3.0
    print(vector[0])
    # CHECK: 4.0
    print(vector[1])
    # CHECK: 5.0
    print(vector[2])
    # CHECK: 6.0
    print(vector[3])
    # CHECK: 7.0
    print(vector[4])

    @parameter
    @__copy_capture(vector)
    fn add(idx: Int):
        vector[idx] = vector[idx] + vector[idx]

    map[add](len(vector))

    # CHECK: 6.0
    print(vector[0])
    # CHECK: 8.0
    print(vector[1])
    # CHECK: 10.0
    print(vector[2])
    # CHECK: 12.0
    print(vector[3])
    # CHECK: 14.0
    print(vector[4])


fn main():
    test_map()
