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

# RUN: not mojo precompile %S/test_package -o test.mojonot 2>&1 | FileCheck %s
# RUN: not mojo precompile %S/test_package -o not-a-directory/ 2>&1 | FileCheck %s
# RUN: not mojo precompile --diagnostic-format json %S/test_package \
# RUN:   -o test.mojonot 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG
# CHECK: output path must have a '.mojoc' extension
# CHECK-DIAG: "kind":"error","message":"output path must have a '.mojoc' extension"}

## Every dotted component of a namespace must be non-empty.
# RUN: not mojo precompile %S/test_package --namespace "" -o %t.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NS-EMPTY
# RUN: not mojo precompile %S/test_package --namespace a..b -o %t.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NS-INNER
# RUN: not mojo precompile %S/test_package --namespace .a -o %t.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NS-LEADING
# RUN: not mojo precompile %S/test_package --namespace a. -o %t.mojoc 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NS-TRAILING
# NS-EMPTY: error: invalid namespace ''
# NS-INNER: error: invalid namespace 'a..b'
# NS-LEADING: error: invalid namespace '.a'
# NS-TRAILING: error: invalid namespace 'a.'
