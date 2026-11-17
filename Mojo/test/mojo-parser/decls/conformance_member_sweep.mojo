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

# RUN: %parse-mojo-isolated -I %S/inputs %s | FileCheck %s

# Checking `name` erases the stub of the trait's default `name`, which sits
# earlier in the decl list than the conformance members. The sweep must still
# reach `other`.

from conformance_member_sweep_input import S


# CHECK-LABEL: lit.struct.decl @S(
# CHECK: kgen.conformance @conformance_member_sweep_input::@Named {
# CHECK-NEXT: kgen.witness "name($0)"
# CHECK-NEXT: kgen.witness "other($0)"
def use(s: S):
    pass
