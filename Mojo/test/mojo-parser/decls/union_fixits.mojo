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

# The fix-its that add a union case's parentheses, in a declaration and in a
# pattern. Uses the JSON diagnostic format to verify each insertion's text and
# position: right after the case name.

# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


__union Maybe:
    # CHECK: "fixIts":[{"end":{"column":14,"line":[[#@LINE+2]]},"start":{"column":14,"line":[[#@LINE+2]]},"text":"()"}]
    # CHECK-SAME: "message":"union case 'none' must have parentheses; add '()' for a case without a payload"
    case none
    case some(Int)


__union Tok:
    case eof()
    case word(String)
    case span(Int, Int)


def patterns(t: Tok):
    match t:
    # CHECK: "fixIts":[{"end":{"column":15,"line":[[#@LINE+2]]},"start":{"column":15,"line":[[#@LINE+2]]},"text":"(_)"}]
    # CHECK-SAME: "message":"pattern for union case 'word' must have parentheses; add '(_)' to match any payload"
    case .word:
        pass
    # CHECK: "fixIts":[{"end":{"column":18,"line":[[#@LINE+2]]},"start":{"column":18,"line":[[#@LINE+2]]},"text":"(_, _)"}]
    # CHECK-SAME: "message":"pattern for union case 'span' must have parentheses; add '(_, _)' to match any payload"
    case Tok.span:
        pass
