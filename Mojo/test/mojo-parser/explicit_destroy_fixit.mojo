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

# Tests the fix-its that move an `@explicit_destroy` message onto the struct's
# `Deinitable` conformance. Uses JSON diagnostics to check exact positions.

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s


# A message moves after the name of a `not Deinitable` entry.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+3]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""},{"end":{"column":28,"line":[[#@LINE+3]]},"start":{"column":28,"line":[[#@LINE+3]]},"text":" else \"use consume()\""}]
# CHECK-SAME: "message":"add 'else \"use consume()\"' to this conformance"
@explicit_destroy("use consume()")
struct OnNot(not Deinitable, Movable):
    pass


# A message moves after the condition of a `where` clause.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+3]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""},{"end":{"column":51,"line":[[#@LINE+3]]},"start":{"column":51,"line":[[#@LINE+3]]},"text":" else \"use deinit_with()\""}]
# CHECK-SAME: "message":"add 'else \"use deinit_with()\"' to this conformance"
@explicit_destroy("use deinit_with()")
struct OnWhere[cond: Bool](Deinitable where (cond), Movable):
    pass


# A multi-line message keeps its source text, in parentheses.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+6]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""},{"end":{"column":32,"line":[[#@LINE+6]]},"start":{"column":32,"line":[[#@LINE+6]]},"text":" else (\"first line\"\n    \" second line\")"}]
# CHECK-SAME: "message":"add 'else \"first line second line\"' to this conformance"
@explicit_destroy(
    "first line"
    " second line"
)
struct MultiLine(not Deinitable):
    pass


# The conformance already has a message, so the fix-it only removes the
# decorator.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+3]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""}]
# CHECK-SAME: "message":"conformance already gives a message; remove '@explicit_destroy'"
@explicit_destroy("")
struct EmptyDecoratorMessage(not Deinitable else "use consume()"):
    pass


# An unlisted `Deinitable` is unconditional, which is an error. The fix-it
# opts out at the start of the conformance list.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+3]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""},{"end":{"column":25,"line":[[#@LINE+3]]},"start":{"column":25,"line":[[#@LINE+3]]},"text":"not Deinitable else \"use consume()\", "}]
# CHECK-SAME: "message":"Add a `not Deinitable` conformance or remove `@explicit_destroy`"
@explicit_destroy("use consume()")
struct UnlistedWithList(Movable):
    pass


# With no conformance list, the fix-it adds one before the `where` clause.
# CHECK: "fixIts":[{"end":{"column":1,"line":[[#@LINE+3]]},"start":{"column":1,"line":[[#@LINE+2]]},"text":""},{"end":{"column":30,"line":[[#@LINE+3]]},"start":{"column":30,"line":[[#@LINE+3]]},"text":"(not Deinitable else \"use consume()\")"}]
# CHECK-SAME: "message":"Add a `not Deinitable` conformance or remove `@explicit_destroy`"
@explicit_destroy("use consume()")
struct UnlistedNoList[n: Int] where n > 0:
    pass


# An empty `()` has no recorded location, so it gets no fix-it.
# CHECK: "fixIts":[],"location":{"column":{{[0-9]+}},"line":[[#@LINE+3]]}
# CHECK-SAME: "message":"Add a `not Deinitable` conformance or remove `@explicit_destroy`"
@explicit_destroy("use consume()")
struct UnlistedEmptyList():
    pass


# Listing `Deinitable` unconditionally contradicts an opt-out, so no fix-it.
# CHECK: "fixIts":[],"location":{"column":{{[0-9]+}},"line":[[#@LINE+3]]}
# CHECK-SAME: "message":"Add a `not Deinitable` conformance or remove `@explicit_destroy`"
@explicit_destroy("use consume()")
struct ListedUnconditional(Deinitable):
    pass
