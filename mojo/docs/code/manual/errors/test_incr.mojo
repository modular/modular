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
from handle_error import incr
from testing import assert_equal, assert_raises


def test_incr_no_error():
    assert_equal(incr(0), 1)
    assert_equal(incr(1), 2)
    assert_equal(incr(Int.MAX - 1), Int.MAX)


def test_incr_error():
    with assert_raises(contains="integer overflow"):
        _ = incr(Int.MAX)


def main():
    test_incr_no_error()
    test_incr_error()
