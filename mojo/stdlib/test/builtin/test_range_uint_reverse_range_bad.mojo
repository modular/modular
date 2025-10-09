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

from testing import TestSuite


# CHECK: test_range_uint_bad_step_size
def test_range_uint_bad_step_size():
    print("== test_range_uint_bad_step_size ==")
    # Ensure constructing a range with a "-1" step size (i.e. reverse range)
    # with UInt is rejected and aborts now via `debug_assert` handler.
    _ = range(UInt(0), UInt(10), UInt(Int(-1)))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
