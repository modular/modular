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


@fieldwise_init
struct MyType(Copyable, Movable):
    var val: Int

    def __bool__(self) -> Bool:
        return self.val > 0

    def __merge_with__[other_type: __type_of(Int)](self) -> Int:
        return Int(self.val)


def main():
    list = [0.5, 1, 2]
    for value in list:
        print(value)

    i = 0
    m = MyType(9)
    print(i if i > 0 else m)  # prints "9"
