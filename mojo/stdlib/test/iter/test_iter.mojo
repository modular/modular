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

from sys.intrinsics import _type_is_eq

from stdlib.collections.list import _ListIter
from testing import TestSuite, assert_true


fn test_iter() raises:
    var l = [1, 2, 3]
    var it = iter(l)
    assert_true(_type_is_eq[__type_of(it), _ListIter[Int, __origin_of(l)]]())
    # Check that iter(iter(l)) is the same as iter(l)
    var _it2 = iter(it)
    assert_true(_type_is_eq[__type_of(_it2), _ListIter[Int, __origin_of(l)]]())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
