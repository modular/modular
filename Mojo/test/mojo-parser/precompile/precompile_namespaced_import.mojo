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

# A package precompiled under an output namespace imports under that
# namespace, alongside another package precompiled into the same one, with
# its types escaping into the importer. It binds only there: mounted at an
# import root or under another namespace, it is rejected.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/ns/inner
# RUN: %precompile-mojo-isolated %S/inputs/namespaced/pkg -namespace ns.inner -o %t.dir/lib/ns/inner/pkg.mojoc
# RUN: %precompile-mojo-isolated %S/inputs/namespaced/sibling -namespace ns.inner -o %t.dir/lib/ns/inner/sibling.mojoc
# RUN: %parse-mojo-isolated -I %t.dir/lib %s | FileCheck %s

# CHECK: !Thing = !lit.struct<@ns::@inner::@pkg::@a::@Thing>
# CHECK: lit.call @ns::@inner::@pkg::@b::@"bfn()"
# CHECK: lit.call tail @ns::@inner::@sibling::@__init__::@"cfn()"()

## A namespace component may share the package's name.
# RUN: mkdir -p %t.dir/same/pkg
# RUN: %precompile-mojo-isolated %S/inputs/namespaced/pkg -namespace pkg -o %t.dir/same/pkg/pkg.mojoc
# RUN: echo "from pkg.pkg.b import bfn" > %t.dir/same.mojo
# RUN: echo "def main():" >> %t.dir/same.mojo
# RUN: echo "    _ = bfn()" >> %t.dir/same.mojo
# RUN: %parse-mojo-isolated -I %t.dir/same %t.dir/same.mojo | FileCheck %s --check-prefix=SAME
# SAME: lit.call @pkg::@pkg::@b::@"bfn()"

## Mounted at an import root under its own name, that package's namespace is
## rejected rather than bound in its place.
# RUN: mkdir -p %t.dir/top-same
# RUN: cp %t.dir/same/pkg/pkg.mojoc %t.dir/top-same/pkg.mojoc
# RUN: echo "from pkg.b import bfn" > %t.dir/top_same.mojo
# RUN: echo "def main():" >> %t.dir/top_same.mojo
# RUN: echo "    _ = bfn()" >> %t.dir/top_same.mojo
# RUN: not %parse-mojo-isolated -I %t.dir/top-same %t.dir/top_same.mojo 2>&1 | FileCheck %s --check-prefix=SHARED-NAME
# SHARED-NAME: error: precompiled package '{{.*}}pkg.mojoc' was built to be imported as 'pkg.pkg', not as 'pkg'

# RUN: mkdir -p %t.dir/top %t.dir/wrong/other
# RUN: cp %t.dir/lib/ns/inner/pkg.mojoc %t.dir/top/pkg.mojoc
# RUN: cp %t.dir/lib/ns/inner/pkg.mojoc %t.dir/wrong/other/pkg.mojoc
# RUN: echo "from pkg.b import bfn" > %t.dir/top.mojo
# RUN: echo "def main():" >> %t.dir/top.mojo
# RUN: echo "    _ = bfn()" >> %t.dir/top.mojo
# RUN: not %parse-mojo-isolated -I %t.dir/top %t.dir/top.mojo 2>&1 | FileCheck %s --check-prefix=TOP
# RUN: echo "from other.pkg.b import bfn" > %t.dir/wrong.mojo
# RUN: echo "def main():" >> %t.dir/wrong.mojo
# RUN: echo "    _ = bfn()" >> %t.dir/wrong.mojo
# RUN: not %parse-mojo-isolated -I %t.dir/wrong %t.dir/wrong.mojo 2>&1 | FileCheck %s --check-prefix=WRONG
# TOP: error: precompiled package '{{.*}}pkg.mojoc' was built to be imported as 'ns.inner.pkg', not as 'pkg'
# WRONG: error: precompiled package '{{.*}}pkg.mojoc' was built to be imported as 'ns.inner.pkg', not as 'other.pkg'

from ns.inner.pkg.a import Thing
from ns.inner.pkg.b import bfn
from ns.inner.sibling import cfn


def main():
    var thing: Thing = bfn()
    _ = cfn()
