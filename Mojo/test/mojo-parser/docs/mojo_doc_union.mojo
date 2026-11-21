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

# RUN: %mojo-docs-isolated --diagnose-missing-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DIAG
# RUN: %mojo-docs-isolated %s | FileCheck %s

"""Checks union constructor documentation and enum value documentation."""

# DIAG-NOT: is missing a doc string
# DIAG: warning: public symbol 'undocumented' is missing a doc string
# DIAG-NOT: is missing a doc string

# CHECK-LABEL: "structs": [
# CHECK: "aliases": [
# CHECK: "signature": "comptime empty",
# CHECK-NEXT: "sinceVersion": "",
# CHECK-NEXT: "summary": "A shape with no area.",
# CHECK: "signature": "comptime undocumented",
# CHECK-NEXT: "sinceVersion": "",
# CHECK-NEXT: "summary": "",
# CHECK: "name": "EnumShape",
# CHECK: "signature": "struct EnumShape",
__enum EnumShape(ImplicitlyCopyable):
    """A shape."""

    case empty
    """A shape with no area."""

    case undocumented


# CHECK: "functions": [
# CHECK: "signature": "def empty() -> Self",
# CHECK-NEXT: "sinceVersion": "",
# CHECK-NEXT: "summary": "A shape with no area."
# CHECK: "signature": "def circle(var payload0: Float64, /) -> Self",
# CHECK-NEXT: "sinceVersion": "",
# CHECK-NEXT: "summary": "A circle with the given radius."
# CHECK: "signature": "def rect(var payload0: Float64, var payload1: Float64, /) -> Self",
# CHECK-NEXT: "sinceVersion": "",
# CHECK-NEXT: "summary": "A rectangle with the given sides."
# CHECK-NOT: undocumented
# CHECK: "name": "UnionShape",
# CHECK: "name": "__Union"
# CHECK: "signature": "struct UnionShape",
__union UnionShape(ImplicitlyCopyable):
    """A shape."""

    case empty()
    """A shape with no area."""

    case circle(Float64)
    """A circle with the given radius."""

    case rect(Float64, Float64)
    """A rectangle with the given sides."""

    case undocumented()
