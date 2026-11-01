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

# Tests the fixit on the transfer-into-borrowed-argument warning: only the '^'
# token is removed, leaving the operand in place. Uses JSON diagnostic format
# to verify the exact fixit positions.

# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s -o /dev/null 2>&1 | FileCheck %s


struct MemType(Movable):
    var field: String

    def __init__(out self):
        self.field = String()

    def peek_self(self):
        pass


def take_imm(imm_arg: String):
    pass


def take_ref(ref ref_arg: String):
    pass


def imm_arg():
    var a = String()
    # CHECK: "fixIts":[{"end":{"column":16,"line":[[#@LINE+2]]},"start":{"column":15,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    take_imm(a^)


def ref_arg():
    var b = String()
    # CHECK: "fixIts":[{"end":{"column":16,"line":[[#@LINE+2]]},"start":{"column":15,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    take_ref(b^)


# The parentheses stay; only the '^' inside them is removed.
def parenthesized():
    var c = String()
    # CHECK: "fixIts":[{"end":{"column":17,"line":[[#@LINE+2]]},"start":{"column":16,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    take_imm((c^))


def field():
    var d = MemType()
    # CHECK: "fixIts":[{"end":{"column":22,"line":[[#@LINE+2]]},"start":{"column":21,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    take_imm(d.field^)


def self_arg():
    var e = MemType()
    # CHECK: "fixIts":[{"end":{"column":7,"line":[[#@LINE+2]]},"start":{"column":6,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    e^.peek_self()


def field_of_transfer():
    var f = MemType()
    # CHECK: "fixIts":[{"end":{"column":11,"line":[[#@LINE+2]]},"start":{"column":10,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    _ = f^.field


def tuple_destructure():
    var t = (String(), String())
    var x: String
    var y: String
    # CHECK: "fixIts":[{"end":{"column":14,"line":[[#@LINE+2]]},"start":{"column":13,"line":[[#@LINE+2]]},"text":""}]
    # CHECK-SAME: "message":"this value is borrowed here, so the transfer has no effect and can be removed"
    x, y = t^
