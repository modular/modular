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
#
# The IR the compiler prints must parse again. This dumps the module just
# before elaboration and round-trips it. The program holds constructs whose
# printed forms did not parse again before.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t && kgen -elaborate %s -mlir-disable-threading \
# RUN:   -mlir-print-ir-before=elaborate-generators -mlir-print-ir-module-scope \
# RUN:   -mlir-print-ir-tree-dir=%t -o /dev/null
# MLIR names the dump after the op it prints, an unnamed `builtin.module`, and
# numbers it in print order, so the one dump is at a fixed path.
# RUN: kgen-opt --verify-roundtrip -o /dev/null \
# RUN:   %t/builtin_module_no-symbol-name/0_elaborate-generators.mlir

from std.atomic import Atomic


def width(n: Int) -> Int:
    return n * 2


def f[n: Int](x: SIMD[DType.uint8, width(n)]) -> Int:
    return Int(x[0]) + n


def load(flag: Atomic[Int]) -> Int:
    return flag.load()


def main():
    var pair = (1, "a")
    print(f[2](SIMD[DType.uint8, width(2)](0)), pair[0])
    var flag = Atomic[Int](3)
    print(load(flag))
