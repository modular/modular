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

# Packages precompiled under an output namespace run when imported under it,
# including one that depends on another in the same namespace. Resolution and
# mount checks are covered by the mojo-parser precompile_namespaced_* tests.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/ns/inner
# RUN: mojo precompile %S/inputs/namespaced_mojoc/pkg --namespace ns.inner -o %t.dir/lib/ns/inner/pkg.mojoc
# RUN: mojo precompile -I %t.dir/lib %S/inputs/namespaced_mojoc/user --namespace ns.inner -o %t.dir/lib/ns/inner/user.mojoc
# RUN: mojo run -I %t.dir/lib %s | FileCheck %s

# CHECK: thing 42
# CHECK-NEXT: thing 7

from ns.inner.pkg.b import bfn
from ns.inner.user import ufn


def main():
    bfn().show()
    ufn().show()
