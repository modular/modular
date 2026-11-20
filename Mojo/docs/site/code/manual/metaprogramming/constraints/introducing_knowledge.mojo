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


# start-struct-knowledge
struct SizedList[size: Int] where size >= 0:
    # Knowledge base:
    # - `size >= 0`
    var data: List[Int]

    def __init__(out self):
        self.data = List[Int](capacity=Self.size)
        # end-struct-knowledge


# start-function-knowledge
def create_sized_list[size: Int]() -> SizedList[size] where size >= 0:
    # Knowledge base:
    # - `size >= 0`
    return SizedList[size]()
    # end-function-knowledge


def describe[size: Int]():
    # start-comptime-if-knowledge
    comptime if size >= 0:
        # Knowledge base:
        # - `size >= 0`

        comptime if size % 2 == 0:
            # Knowledge base:
            # - `size >= 0`
            # - `size % 2 == 0`
            print("nonnegative and even")
        else:
            print("nonnegative and odd")
        # end-comptime-if-knowledge


def build[size: Int]():
    # start-comptime-assert-knowledge
    comptime assert size >= 0
    # Knowledge base:
    # - `size >= 0`

    var l = create_sized_list[size]()
    # end-comptime-assert-knowledge
    print(len(l.data))


def main():
    build[4]()
    describe[4]()
