# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.ffi import external_call
from std.testing import (
    TestSuite,
    assert_false,
)


struct RegisterPassablePointer(RegisterPassable):
    var pointer: OpaquePointer[ExternalOrigin[mut=True]]


def test_external_call_handles_rp_return_types() raises:
    var path = "/does/not/exist/here/file.file"
    var mode = "r"
    var result = external_call["fopen", RegisterPassablePointer](
        path.as_c_string_slice(), mode.as_c_string_slice()
    )
    assert_false(result.pointer._is_not_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
