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
from testing import assert_false, assert_true


fn test_bool_value() raises:
    var session = InferenceSession()

    var false_value = session.new_bool_value(False)
    assert_false(false_value.as_bool())

    var true_value = session.new_bool_value(True)
    assert_true(true_value.as_bool())


fn main() raises:
    test_bool_value()
