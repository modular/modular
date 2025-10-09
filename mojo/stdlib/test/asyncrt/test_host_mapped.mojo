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

from asyncrt_test_utils import create_test_device_context, expect_eq
from testing import TestSuite


def test_host_mapped():
    var ctx = create_test_device_context()

    alias length = 20

    var in_buf = ctx.enqueue_create_buffer[DType.int64](length)
    var out_buf = ctx.enqueue_create_buffer[DType.int64](length)

    with in_buf.map_to_host() as in_map:
        for i in range(length):
            in_map[i] = i

    in_buf.enqueue_copy_to(out_buf)

    with out_buf.map_to_host() as out_map:
        for i in range(length):
            expect_eq(out_map[i], i)

    print("Done")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
