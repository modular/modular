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

from my_math import inc
from testing import assert_equal, assert_raises


def test_inc_valid():
    assert_equal(inc(0), 1)
    assert_equal(inc(1), 2)


def test_inc_max():
    with assert_raises():
        # Assign the return value to the discard pattern to prevent the Mojo
        # compiler from warning that it is unused.
        _ = inc(Int.MAX)
