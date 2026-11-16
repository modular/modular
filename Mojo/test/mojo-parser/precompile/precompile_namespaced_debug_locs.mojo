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

# A package precompiled under an output namespace is nested in a namespace per
# component, and its debug-info source names and rendered type strings root at
# the namespace rather than at the package.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg %t.dir/out/ns
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "trait Nameable:" > %t.dir/pkg/t.mojo
# RUN: echo "    def name(self) -> Int:" >> %t.dir/pkg/t.mojo
# RUN: echo "        ..." >> %t.dir/pkg/t.mojo
# RUN: echo "" >> %t.dir/pkg/t.mojo
# RUN: echo "struct Holder[T: Nameable]:" >> %t.dir/pkg/t.mojo
# RUN: echo "    def get(self) -> Int:" >> %t.dir/pkg/t.mojo
# RUN: echo "        return 0" >> %t.dir/pkg/t.mojo
# RUN: %precompile-mojo-isolated %t.dir/pkg -namespace ns -o %t.dir/out/ns/pkg.mojoc
# RUN: kgen-opt --mlir-print-debuginfo %t.dir/out/ns/pkg.mojoc \
# RUN:   | FileCheck %s --implicit-check-not 'source_name<(pkg)"pkg">' \
# RUN:       --implicit-check-not '<@pkg::'

# CHECK-DAG: #[[PKG:.+]] = #debuginfo.source_name<(pkg)"pkg" from <(pkg)"ns">>
# CHECK-DAG: (module)"a" from #[[PKG]]>
# CHECK-DAG: (struct)"Holder"[<"trait<@ns::@pkg::@t::@Nameable
# CHECK-DAG: {{^  }}lit.package namespace @ns {
# CHECK-DAG: {{^    }}lit.package @pkg {
