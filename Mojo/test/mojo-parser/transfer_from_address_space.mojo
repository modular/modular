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
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

# A transfer out of a non-default address space copies a memory-only, trivially
# copyable value into the default address space with a memcpy. The original
# ends there.


struct MemTC(ImplicitlyCopyable):
    var value: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0


def take_owned(var x: MemTC):
    pass


def take_ref[addr_space: AddressSpace](ref[_, addr_space] x: MemTC):
    pass


# CHECK-LABEL: lit.fn @"transfer_to_owned_argument
# CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemTC, mut *"s`", sugar_preserved({{.*}}, 2)>
# CHECK: [[TMP:%.*]] = lit.var.decl "anonymous
# CHECK-NEXT: lit.memcpy [[SRC:%.*]], [[TMP]] : <!MemTC, mut *"s`", sugar_preserved({{.*}}, 2)> -> <!MemTC, mut *"anonymous
# CHECK-NEXT: lit.ownership.mark_destroyed [[SRC]]
# CHECK-NEXT: lit.call {{.*}}@"take_owned{{.*}}([[TMP]])
def transfer_to_owned_argument():
    @__address_space(.SHARED)
    var s = MemTC()
    take_owned(s^)


# CHECK-LABEL: lit.fn @"transfer_to_result
# CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemTC, mut *"s`{{[0-9]*}}", sugar_preserved({{.*}}, 2)>
# CHECK: [[TMP:%.*]] = lit.var.decl "anonymous
# CHECK-NEXT: lit.memcpy [[SRC:%.*]], [[TMP]] : <!MemTC, mut *"s`{{[0-9]*}}", sugar_preserved({{.*}}, 2)> -> <!MemTC, mut *"anonymous
# CHECK-NEXT: lit.ownership.mark_destroyed [[SRC]]
# CHECK: lit.call {{.*}}MemTC::@"__init__(move:{{.*}}([[TMP]],
def transfer_to_result() -> MemTC:
    @__address_space(.SHARED)
    var s = MemTC()
    return s^


def use_after_transfer():
    @__address_space(.SHARED)
    var s = MemTC()  # expected-note {{'s' declared here}}
    take_owned(s^)
    take_ref(s)  # expected-error {{use of uninitialized value 's'}}
