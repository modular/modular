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
# RUN: mkdir -p %t.closure-dir
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 mojo precompile %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 mojo -debug-level=line-tables -I %t.closure-dir %s 4 | FileCheck %s
# RUN: mojo precompile %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: mojo -debug-level=line-tables -I %t.closure-dir %s 4 | FileCheck %s

from std.sys import argv
from closure import emitLoad


def main() raises:
    var x = Int(atol(argv()[1]))
    # CHECK: 4
    emitLoad(x)
