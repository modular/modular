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

# While precompiling, a package's self-imports spelled by the path it will be
# imported by resolve to its source, never to what is on disk under that path:
# here the output being written, which sits on the import path. The bare
# source directory name names the same package, so the two spellings mix.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/ns/inner %t.dir/empty %t.dir/other
# RUN: %precompile-mojo-isolated -I %t.dir/lib %S/inputs/namespaced/selfpkg -namespace ns.inner -o %t.dir/lib/ns/inner/selfpkg.mojoc
# RUN: %parse-mojo-isolated -I %t.dir/lib %s | FileCheck %s
# CHECK: lit.call @ns::@inner::@selfpkg::@c::@"roundtrip()"

## The namespace need not exist under any import root while precompiling.
# RUN: %precompile-mojo-isolated -I %t.dir/empty %S/inputs/namespaced/selfpkg -namespace ns.inner -o %t.dir/other/selfpkg.mojoc

## A plain rename binds its output name the same way.
# RUN: mkdir -p %t.dir/src/pkg %t.dir/rlib
# RUN: echo "# pkg" > %t.dir/src/pkg/__init__.mojo
# RUN: echo "def afn() -> Int:" > %t.dir/src/pkg/a.mojo
# RUN: echo "    return 7" >> %t.dir/src/pkg/a.mojo
# RUN: echo "from renamed.a import afn" > %t.dir/src/pkg/b.mojo
# RUN: echo "def bfn() -> Int:" >> %t.dir/src/pkg/b.mojo
# RUN: echo "    return afn()" >> %t.dir/src/pkg/b.mojo
# RUN: %precompile-mojo-isolated -I %t.dir/rlib %t.dir/src/pkg -o %t.dir/rlib/renamed.mojoc
# RUN: echo "from renamed.b import bfn" > %t.dir/renamed_main.mojo
# RUN: echo "def main():" >> %t.dir/renamed_main.mojo
# RUN: echo "    _ = bfn()" >> %t.dir/renamed_main.mojo
# RUN: %parse-mojo-isolated -I %t.dir/rlib %t.dir/renamed_main.mojo | FileCheck %s --check-prefix=RENAMED
# RENAMED: lit.call tail @renamed::@b::@"bfn()"()

from ns.inner.selfpkg.c import roundtrip


def main():
    _ = roundtrip()
