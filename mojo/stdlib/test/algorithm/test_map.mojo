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
from testing import assert_equal
from testing import TestSuite


def test_map():
    var vector_stack = InlineArray[Float32, 5](1.0, 2.0, 3.0, 4.0, 5.0)
    var vector = NDBuffer[DType.float32, 1, _, 5](vector_stack)

    @parameter
    @__copy_capture(vector)
    fn add_two(idx: Int):
        vector[idx] = vector[idx] + 2

    map[add_two](len(vector))

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    @parameter
    @__copy_capture(vector)
    fn add(idx: Int):
        vector[idx] = vector[idx] + vector[idx]

    map[add](len(vector))

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def main():
    var suite = TestSuite()

    suite.test[test_map]()

    suite^.run()
