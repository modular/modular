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

from random import rand

from testing import TestSuite


fn _sort_test[dtype: DType, name: StaticString](size: Int, max: Int) raises:
    var p = UnsafePointer[Scalar[dtype]].alloc(size)
    rand[dtype](p, size)
    sort(Span[Scalar[dtype], MutableAnyOrigin](ptr=p, length=UInt(size)))
    for i in range(1, size - 1):
        if p[i] < p[i - 1]:
            print(name, "size:", size, "max:", max, "incorrect sort")
            print("p[", end="")
            print(i - 1, end="")
            print("] =", p[i - 1])
            print("p[", end="")
            print(i, end="")
            print("] =", p[i])
            print()
            p.free()
            raise Error("Failed")
    p.free()


def test_sort_issue_1018():
    _sort_test[DType.int8, "int8"](300, 3_000)
    _sort_test[DType.float32, "float32"](3_000, 3_000)
    _sort_test[DType.float64, "float64"](300_000, 3_000_000_000)


def main():
    var suite = TestSuite()

    suite.test[test_sort_issue_1018]()
    suite^.run()
