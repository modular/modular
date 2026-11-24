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

# A function signature loaded from bytecode can call a function whose body
# calls the first function again. `Holder.__init__` takes `Flag[pick()]`, and
# `pick` constructs a `Holder`. Resolving that signature fully resolves `pick`,
# and the walk over `pick`'s body must not report a recursive reference to the
# `Holder.__init__` whose signature is still being resolved.
#
# Reduced from `CStringSpan("hi")` against a precompiled std, where
# `CStringSpan.__init__` takes `Pointer[c_char, ...]`, `c_char` calls
# `_c_char_dtype()`, and that function's callees reach
# `StringSpan.as_c_string_span`, which calls the same `CStringSpan.__init__`.

# RUN: rm -rf %t.signature-body-cycle
# RUN: mkdir -p %t.signature-body-cycle
# RUN: %precompile-mojo-isolated %S/inputs/signature_body_cycle_package -o %t.signature-body-cycle/signature_body_cycle_package.mojoc
# RUN: %parse-mojo-isolated -I %t.signature-body-cycle %s | FileCheck %s

from signature_body_cycle_package import Flag, Holder, pick


# CHECK-LABEL: lit.fn @"test()"
# CHECK: lit.call @signature_body_cycle_package::@__init__::@Holder::@"__init__(
def test():
    _ = Holder(Flag[pick()]())
