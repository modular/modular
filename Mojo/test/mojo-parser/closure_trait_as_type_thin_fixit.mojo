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

# Tests that using a non-`thin` `def(...)` closure trait where a type is
# expected explains the mistake and offers a fix-it inserting `thin` after the
# argument list. Uses JSON diagnostic format to verify the exact fix-it position.

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


struct Holder[T: Movable]:
    pass


def main():
    # CHECK: "message":"'Holder' parameter 'T' has 'Movable' type, but value is the closure trait 'def(Int) -> Int', not a type"
    # CHECK: "fixIts":[{"end":{"column":28,"line":[[#@LINE+2]]},"start":{"column":28,"line":[[#@LINE+2]]},"text":" thin"}]
    # CHECK-SAME: "message":"a 'def' type without 'thin' is a closure trait; add 'thin' after the argument list to make it a function pointer type"
    var h = Holder[def(Int) -> Int]()
