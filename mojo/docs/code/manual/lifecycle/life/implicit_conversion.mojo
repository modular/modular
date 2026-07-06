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
from std.testing import assert_equal


struct Source(Copyable):
    def __init__(out self):
        pass


struct Target(Copyable, Writable):
    @implicit
    def __init__(out self, s: Source):
        pass


def main() raises:
    # start-implicit-conversion
    var limit: Optional[Int] = None
    # end-implicit-conversion
    assert_equal(limit, None)

    var name1: String = "Sam"
    var name2 = String("Sam")
    var name3 = "Sam"
    assert_equal(name1, name2)
    assert_equal(name3, "Sam")

    var a = Source()
    var b: Target = a
    print(b)
    _ = b
