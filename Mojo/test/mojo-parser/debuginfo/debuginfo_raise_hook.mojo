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

# RUN: %parse-mojo-isolated -debug-level full -O0 -mlir-print-debuginfo %s | FileCheck %s

# CHECK-LABEL: lit.fn @"foo()"
# CHECK: lit.call tail @std::@builtin::@error::@"__mojo_debugger_raise_hook()"()
# CHECK-NEXT: debuginfo.line_table_loc
# CHECK-NEXT: lit.raise
def foo() raises:
    raise "exception"


# A bare `raise` has no error expression for the hook call to use.
# CHECK-LABEL: lit.fn @"bar()"
# CHECK: lit.call tail @std::@builtin::@error::@"__mojo_debugger_raise_hook()"()
# CHECK-NEXT: debuginfo.line_table_loc
# CHECK-NEXT: lit.raise
def bar() raises:
    try:
        foo()
    except:
        raise
