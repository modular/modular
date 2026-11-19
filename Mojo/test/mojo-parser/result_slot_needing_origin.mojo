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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# An initializer with a parametric result address space or origin needs memory
# for its result before the parameters can be inferred. When the destination
# has no memory, as for an argument or a setter, the result goes into a
# temporary first.


struct CopyType(ImplicitlyCopyable):
    var value: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

    def __init__(a: Int, out[_] self):
        self.value = a


def take_owned(var value: CopyType):
    _ = value^


struct Dynamic:
    def __getattr__(self, name: StringLiteral) -> CopyType:
        return CopyType()

    def __setattr__(self, name: StringLiteral, var value: CopyType):
        _ = value^


# CHECK-LABEL: lit.fn @"pass_temporary_with_parametric_address_space()"
# CHECK-NEXT: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: lit.call {{.*}}CopyType::@"__init__
# CHECK-SAME: <:!AddressSpace {_value: !SIMDLength = {0}}, :origin<true> *"anonymous*{{.*}}">(%[[TMP]])
# CHECK-NEXT: %[[COPY:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: lit.memcpy %[[TMP]], %[[COPY]]
# CHECK-NEXT: lit.call {{.*}}@"take_owned{{.*}}(%[[COPY]])
def pass_temporary_with_parametric_address_space():
    take_owned(CopyType())


# CHECK-LABEL: lit.fn @"pass_temporary_with_parametric_origin()"
# CHECK-NEXT: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: kgen.param.constant
# CHECK-NEXT: lit.call {{.*}}CopyType::@"__init__
# CHECK-SAME: <:origin<true> *"anonymous*{{.*}}">({{.*}}, %[[TMP]])
# CHECK-NEXT: %[[COPY:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: lit.memcpy %[[TMP]], %[[COPY]]
# CHECK-NEXT: lit.call {{.*}}@"take_owned{{.*}}(%[[COPY]])
def pass_temporary_with_parametric_origin():
    take_owned(CopyType(0))


# CHECK-LABEL: lit.fn @"assign_temporary_with_parametric_address_space
# CHECK-NEXT: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: lit.call {{.*}}CopyType::@"__init__
# CHECK-SAME: <:!AddressSpace {_value: !SIMDLength = {0}}, :origin<true> *"anonymous*{{.*}}">(%[[TMP]])
# CHECK-NEXT: kgen.param.constant: none
# CHECK-NEXT: lit.return
def assign_temporary_with_parametric_address_space(d: Dynamic):
    d.value = CopyType()


# CHECK-LABEL: lit.fn @"assign_temporary_with_parametric_origin
# CHECK-NEXT: %[[TMP:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!CopyType
# CHECK-NEXT: kgen.param.constant
# CHECK-NEXT: lit.call {{.*}}CopyType::@"__init__
# CHECK-SAME: <:origin<true> *"anonymous*{{.*}}">({{.*}}, %[[TMP]])
# CHECK-NEXT: kgen.param.constant: none
# CHECK-NEXT: lit.return
def assign_temporary_with_parametric_origin(d: Dynamic):
    d.value = CopyType(0)
