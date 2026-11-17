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

# A package precompiled against another package within a namespace records
# that dependency by its full import path, never by the namespace, and the
# dependency loads from there when the package is imported. Both packages
# share the namespace, and a type from one escapes through the other.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/ns/inner
# RUN: %precompile-mojo-isolated %S/inputs/namespaced/pkg -namespace ns.inner -o %t.dir/lib/ns/inner/pkg.mojoc
# RUN: %precompile-mojo-isolated -I %t.dir/lib %S/inputs/namespaced/user -namespace ns.inner -o %t.dir/lib/ns/inner/user.mojoc
# RUN: kgen-opt %t.dir/lib/ns/inner/user.mojoc | FileCheck %s --check-prefix=DEPS
# RUN: %parse-mojo-isolated -I %t.dir/lib %s | FileCheck %s

# DEPS: lit.package @user {
# DEPS: } <dependencies = [@ns::@inner::@pkg, @std]>

# CHECK: !Thing = !lit.struct<@ns::@inner::@pkg::@a::@Thing>
# CHECK: lit.call @ns::@inner::@user::@__init__::@"ufn()"

from ns.inner.user import ufn


def main():
    _ = ufn()
