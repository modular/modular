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
struct MyInt(Comparable):
    var value: Int

    fn __eq__(self, rhs: MyInt) -> Bool:
        return self.value == rhs.value

    fn __ne__(self, rhs: MyInt) -> Bool:
        return self.value != rhs.value

    fn __lt__(self, rhs: MyInt) -> Bool:
        return self.value < rhs.value

    fn __le__(self, rhs: MyInt) -> Bool:
        return self.value <= rhs.value

    fn __gt__(self, rhs: MyInt) -> Bool:
        return self.value > rhs.value

    fn __ge__(self, rhs: MyInt) -> Bool:
        return self.value >= rhs.value


def main():
    m1 = MyInt(7)
    m2 = MyInt(5)
    print(m1 == m2)
    print(m1 != m2)
    print(m1 < m2)
    print(m1 <= m2)
    print(m1 > m2)
    print(m1 >= m2)
