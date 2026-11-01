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

# `kgen -precompile` packages a Mojo source directory the way `mojo
# precompile` does, against only the import roots it is given, so
# precompilation is testable without the real standard library.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg %t.dir/user %t.dir/deps
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "import pkg.a" > %t.dir/pkg/b.mojo
# RUN: echo "def bfn():" >> %t.dir/pkg/b.mojo
# RUN: echo "    pkg.a.afn()" >> %t.dir/pkg/b.mojo
# RUN: echo "# user" > %t.dir/user/__init__.mojo
# RUN: echo "from dep.a import afn" > %t.dir/user/u.mojo
# RUN: echo "def ufn():" >> %t.dir/user/u.mojo
# RUN: echo "    afn()" >> %t.dir/user/u.mojo

## With -S, the package prints under its directory's name. The auto-imported
## builtin package comes from the fixture std on the include path.
# RUN: %precompile-mojo-isolated -S %t.dir/pkg \
# RUN:   | FileCheck %s --check-prefix=PRINT
# PRINT:      lit.package @pkg {
# PRINT-NOT:  dependencies
# PRINT:      lit.import @std path <0, ["std"]>
# PRINT:      lit.import @pkg path <0, ["pkg"]>
# PRINT:      lit.call tail @pkg::@a::@"afn()"()

## Written to a `.mojoc` file, the package takes the file's name.
# RUN: %precompile-mojo-isolated %t.dir/pkg -o %t.dir/deps/dep.mojoc
# RUN: kgen-opt %t.dir/deps/dep.mojoc \
# RUN:   | FileCheck %s --check-prefix=RENAMED --implicit-check-not '@pkg::'
# RENAMED:      lit.package @dep {
# RENAMED:      lit.import @pkg path <0, ["dep"]>
# RENAMED:      lit.call tail @dep::@a::@"afn()"()

## A precompiled package imported by the package under compilation is
## recorded as its link dependency.
# RUN: %precompile-mojo-isolated -S -I %t.dir/deps %t.dir/user \
# RUN:   | FileCheck %s --check-prefix=DEPS
# DEPS: lit.package @user attributes {dependencies = #kgen<link.dependencies[@dep]>}

# RUN: not kgen -precompile %t.dir/user/u.mojo 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NOT-PKG
# NOT-PKG: '{{.*}}u.mojo' does not correspond to a Mojo package
