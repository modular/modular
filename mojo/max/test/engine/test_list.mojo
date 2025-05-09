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
# RUN: %mojo -debug-level full %s

from max.engine import InferenceSession
from testing import assert_equal, assert_false, assert_true


fn test_list_value() raises:
    var session = InferenceSession()
    var list_value = session.new_list_value()
    var list = list_value.as_list()

    assert_equal(len(list), 0)

    var false_value = session.new_bool_value(False)
    var true_value = session.new_bool_value(True)

    list.append(false_value)
    assert_equal(len(list), 1)

    list.append(true_value)
    assert_equal(len(list), 2)

    assert_false(list[0].as_bool())
    assert_true(list[1].as_bool())

    _ = list_value^


fn main() raises:
    test_list_value()
