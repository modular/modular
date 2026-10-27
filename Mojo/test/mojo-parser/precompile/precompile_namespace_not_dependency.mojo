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

# A namespace only groups packages, so a precompiled package never records one
# as a link dependency. Each package within a namespace, or a namespace nested
# in one, that it imports is recorded by its full import path, and is imported
# by it when the dependency is resolved; a sibling package or namespace it never
# imports is not recorded.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/ns/sub %t.dir/src/ns/inner/utils %t.dir/src/ns/unused %t.dir/src/ns/tools %t.dir/pkg %t.dir/lib
# RUN: echo "def toolfn() -> Int:" > %t.dir/src/ns/tools/t.mojo
# RUN: echo "    return 9" >> %t.dir/src/ns/tools/t.mojo
# RUN: echo "def helper() -> Int:" > %t.dir/src/ns/helper.mojo
# RUN: echo "    return 5" >> %t.dir/src/ns/helper.mojo
# RUN: echo "def subfn() -> Int:" > %t.dir/src/ns/sub/__init__.mojo
# RUN: echo "    return 6" >> %t.dir/src/ns/sub/__init__.mojo
# RUN: echo "def utilsfn() -> Int:" > %t.dir/src/ns/inner/utils/__init__.mojo
# RUN: echo "    return 7" >> %t.dir/src/ns/inner/utils/__init__.mojo
# RUN: echo "def unusedfn() -> Int:" > %t.dir/src/ns/unused/__init__.mojo
# RUN: echo "    return 8" >> %t.dir/src/ns/unused/__init__.mojo
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "from ns.helper import helper" > %t.dir/pkg/p.mojo
# RUN: echo "from ns.sub import subfn" >> %t.dir/pkg/p.mojo
# RUN: echo "from ns.inner.utils import utilsfn" >> %t.dir/pkg/p.mojo
# RUN: echo "def pfn() -> Int:" >> %t.dir/pkg/p.mojo
# RUN: echo "    return helper() + subfn() + utilsfn()" >> %t.dir/pkg/p.mojo
# RUN: %precompile-mojo-isolated -I %t.dir/src %t.dir/pkg -o %t.dir/lib/pkg.mojoc
# RUN: kgen-opt %t.dir/lib/pkg.mojoc | FileCheck %s --check-prefix=DEPS
# RUN: %parse-mojo-isolated -I=%t.dir/lib -I=%t.dir/src %s | FileCheck %s

# DEPS: lit.package @pkg attributes {dependencies = #kgen<link.dependencies[@std, @ns::@inner::@utils, @ns::@sub]>}

# CHECK: lit.call {{.*}}@pkg::@p::@"pfn()"

from pkg.p import pfn


def main():
    _ = pfn()
