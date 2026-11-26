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

# A package documented under an output namespace resolves its own imports
# spelled with that namespace, and its doc paths include it.
# RUN: mojo doc %S/inputs/namespaced/pkg -namespace ns -o %t.json
# RUN: FileCheck %s --input-file %t.json
# CHECK: "path": "/ns/pkg/a/Thing"
# CHECK-NOT: "path": "/pkg/a/Thing"

## Without the namespace the package can't import itself under it.
# RUN: not mojo doc %S/inputs/namespaced/pkg -o %t.no-ns.json 2>&1 | FileCheck %s --check-prefix=NO-NS
# NO-NS: error: unable to locate module 'ns'

## A namespace may not have empty components.
# RUN: not mojo doc %S/inputs/namespaced/pkg -namespace ns..bad -o %t.bad.json 2>&1 | FileCheck %s --check-prefix=BAD
# BAD: invalid namespace 'ns..bad'
