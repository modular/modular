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
struct MyInt:
    var value: Int

    def __add__(self, rhs: MyInt) -> Self:
        return MyInt(self.value + rhs.value)

    def __add__(self, rhs: Int) -> Self:
        return MyInt(self.value + rhs)

    def __radd__(self, lhs: Int) -> Self:
        return MyInt(self.value + lhs)

    def __iadd__(mut self, rhs: MyInt) -> None:
        self.value += rhs.value

    def __iadd__(mut self, rhs: Int) -> None:
        self.value += rhs


def main():
    m1 = MyInt(5)
    m2 = MyInt(2)
    m3 = m1 + m2
    m4 = 6 + m1
    m5 = m1 + 6
    print(m3.value, m4.value, m5.value)  # 7 11 11
    m4 += 1
    m4 += m1
    print(m4.value)  # 17
