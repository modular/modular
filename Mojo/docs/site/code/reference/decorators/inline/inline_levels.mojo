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

from std.testing import *


@inline(.always)
def scale(x: Int) -> Int:
    return x * 2


@inline(.nodebug)
def offset(x: Int) -> Int:
    return x + 1


@inline(.never)
def cold_path(x: Int) -> Int:
    return x - 1


@inline(.automatic)
def compiler_decides(x: Int) -> Int:
    return x


def main() raises:
    assert_equal(scale(4), 8)
    assert_equal(offset(4), 5)
    assert_equal(cold_path(4), 3)
    assert_equal(compiler_decides(4), 4)
