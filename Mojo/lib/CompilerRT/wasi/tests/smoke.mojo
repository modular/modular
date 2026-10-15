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
from std.math import sqrt, floor, ceil
from std.collections import List, Dict


def main():
    # Strings
    var s = "Hello, wasm64!"
    print(s)
    print("upper:", s.upper())
    print("byte_length:", s.byte_length())
    print("starts_with:", s.startswith("Hello"))

    # Loops
    var total = 0
    for i in range(1, 11):
        total += i
    print("sum 1..10:", total)

    # Math
    print("sqrt(2):", sqrt(2.0))
    print("floor(2.7):", floor(2.7))
    print("ceil(2.1):", ceil(2.1))

    # List
    var xs = [1, 2, 3, 4]
    print("list:", xs)
    print("list len:", len(xs))
    var list_total = 0
    for x in xs:
        list_total += x
    print("list sum:", list_total)

    # Dict
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    print("dict:", d)
    print("dict a:", d.get("a", -1))
