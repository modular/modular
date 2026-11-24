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

# A package can only be precompiled into a namespace: every component of its
# output namespace that an import root or the package's working directory
# already provides must be a plain directory, never a source package,
# precompiled package or module.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg %t.dir/src-pkg/ns %t.dir/src-nested/ns/inner %t.dir/src-mojoc %t.dir/src-module
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo

## A source package at the namespace's root.
# RUN: echo "# ns" > %t.dir/src-pkg/ns/__init__.mojo
# RUN: not %precompile-mojo-isolated -I %t.dir/src-pkg %t.dir/pkg -namespace ns.inner -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=PKG
# PKG: error: cannot precompile into namespace 'ns.inner': 'ns' names '{{.*}}src-pkg{{/|\\}}ns', not a namespace

## A source package nested in a namespace.
# RUN: echo "# inner" > %t.dir/src-nested/ns/inner/__init__.mojo
# RUN: not %precompile-mojo-isolated -I %t.dir/src-nested %t.dir/pkg -namespace ns.inner -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NESTED
# NESTED: error: cannot precompile into namespace 'ns.inner': 'ns.inner' names '{{.*}}src-nested{{/|\\}}ns{{/|\\}}inner', not a namespace

## A precompiled package.
# RUN: %precompile-mojo-isolated %t.dir/pkg -o %t.dir/src-mojoc/ns.mojoc
# RUN: not %precompile-mojo-isolated -I %t.dir/src-mojoc %t.dir/pkg -namespace ns -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=MOJOC
# MOJOC: error: cannot precompile into namespace 'ns': 'ns' names '{{.*}}ns.mojoc', not a namespace

## A module.
# RUN: echo "def nfn():" > %t.dir/src-module/ns.mojo
# RUN: echo "    pass" >> %t.dir/src-module/ns.mojo
# RUN: not %precompile-mojo-isolated -I %t.dir/src-module %t.dir/pkg -namespace ns -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=MODULE
# MODULE: error: cannot precompile into namespace 'ns': 'ns' names '{{.*}}ns.mojo', not a namespace

## A module beside the package, reached only through the working directory
## that the package's own imports search.
# RUN: mkdir -p %t.dir/wd/pkg %t.dir/wd-nested/pkg %t.dir/wd-nested/ns
# RUN: echo "# pkg" > %t.dir/wd/pkg/__init__.mojo
# RUN: echo "def nfn():" > %t.dir/wd/ns.mojo
# RUN: echo "    pass" >> %t.dir/wd/ns.mojo
# RUN: not %precompile-mojo-isolated %t.dir/wd/pkg -namespace ns -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=WD
# WD: error: cannot precompile into namespace 'ns': 'ns' names '{{.*}}wd{{/|\\}}ns.mojo', not a namespace

# RUN: echo "# pkg" > %t.dir/wd-nested/pkg/__init__.mojo
# RUN: echo "def nfn():" > %t.dir/wd-nested/ns/inner.mojo
# RUN: echo "    pass" >> %t.dir/wd-nested/ns/inner.mojo
# RUN: not %precompile-mojo-isolated %t.dir/wd-nested/pkg -namespace ns.inner -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=WD-NESTED
# WD-NESTED: error: cannot precompile into namespace 'ns.inner': 'ns.inner' names '{{.*}}wd-nested{{/|\\}}ns{{/|\\}}inner.mojo', not a namespace

## The standard library is a package too.
# RUN: not %precompile-mojo-isolated %t.dir/pkg -namespace std.ns -o %t.dir/out.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=STD
# STD: error: cannot precompile into namespace 'std.ns': 'std'{{.*}}, not a namespace
