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

# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo \
# RUN:   | kgen-opt -lower-semantic-cf -verify-diagnostics

# Guard bindings are declared before the match so the successful binding remains
# available to the continuation.
# CHECK-LABEL: lit.fn @"guard_binding
# CHECK: [[VALUE:%.*]] = lit.var.decl "value" var
# CHECK: hlcf.match
# CHECK: lit.ref.store %x, [[VALUE]]
# CHECK: hlcf.match.complete
# CHECK: ([[VALUE]],
def guard_binding(x: Int):
    __guard case var value = x else:
        return  # expected-warning {{unreachable code after match that always completes}}
    value += 1


# The failed guard path cannot fall through into the continuation.
# CHECK-LABEL: lit.fn @"guard_no_fallthrough
# CHECK: hlcf.match
# CHECK: else {
# CHECK: hlcf.unreachable <fallthroughError = "guard 'else' block must not fall through">
def guard_no_fallthrough(x: Int):
    __guard case 0 = x else:  # expected-error {{guard 'else' block must not fall through}}
        pass
    _ = x


# CHECK-LABEL: lit.fn @"comptime_guard_binding
# CHECK: hlcf.comptime.if
# CHECK: hlcf.comptime.yield
# CHECK: else {
# CHECK: lit.return
# CHECK: }
def comptime_guard_binding[pair: Tuple[Int, Int]]() -> Int:
    comptime __guard case (0, value) = pair else:
        return -1
    return value
