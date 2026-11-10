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
"""Runs the public `rowwise.reduce` bodies on GPU."""

from max.gpu.host import DeviceContext
from rowwise_reduce_test_utils.public_reduce import check_doubled, check_sums
from std.testing import TestSuite


def test_state_carrying() raises:
    with DeviceContext() as ctx:
        check_sums(ctx, 4096, 1024)
        check_sums(ctx, 64, 1000)
        check_sums(ctx, 7, 37)


def test_stateless() raises:
    with DeviceContext() as ctx:
        check_doubled(ctx, 4096, 1024)
        check_doubled(ctx, 64, 1000)
        check_doubled(ctx, 7, 37)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
