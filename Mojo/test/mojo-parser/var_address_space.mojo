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


struct MemType(Movable):
    var value: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0


# CHECK-LABEL: lit.fn @"explicit_type
# CHECK: %x = lit.var.decl "x" var : !lit.ref<{{.*}}Int, mut *"x`", sugar_preserved(#lit.struct.extract<{{.*}}#kgen.type<!AddressSpace>, "_value">, "_mlir_value">, 2)>
# CHECK: lit.ref.store %{{.*}}, %x
def explicit_type():
    @__address_space(AddressSpace.SHARED)
    var x: Int = 0


# CHECK-LABEL: lit.fn @"inferred_type
# CHECK: %x = lit.var.decl "x" var : !lit.ref<{{.*}}Int, mut *"x`", sugar_preserved({{.*}}, 2)>
def inferred_type():
    @__address_space(.SHARED)
    var x = 1


# CHECK-LABEL: lit.fn @"no_initializer
# CHECK: %x = lit.var.decl "x" var : !lit.ref<{{.*}}Int, mut *"x`", sugar_preserved({{.*}}, 4)>
def no_initializer():
    @__address_space(.LOCAL)
    var x: Int


# CHECK-LABEL: lit.fn @"parametric
# CHECK: %x = lit.var.decl "x" var : !lit.ref<{{.*}}Int, mut *"x`", #lit.struct.extract<:!SIMDLength #lit.struct.extract<:!AddressSpace addr_space, "_value">, "_mlir_value">>
def parametric[addr_space: AddressSpace]():
    @__address_space(addr_space)
    var x: Int = 0


# The initializer's address space parameter is inferred from the var.
# CHECK-LABEL: lit.fn @"infer_out_address_space
# CHECK: %m = lit.var.decl "m" var : !lit.ref<!MemType, mut *"m`", sugar_preserved({{.*}}, 2)>
# CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
# CHECK-SAME: <:!AddressSpace {{.*}}, 2)}}, :origin<true> *"m`">(%m)
def infer_out_address_space():
    @__address_space(.SHARED)
    var m = MemType()


# CHECK-LABEL: lit.fn @"undecorated
# CHECK: %x = lit.var.decl "x" var : !lit.ref<{{.*}}Int, mut *"x`">
def undecorated():
    var x: Int = 0


# CHECK-LABEL: lit.fn @"match_default_value
# CHECK-NOT: lit.var.decl
# CHECK: lit.ref.struct.ger %m[value] : <!MemType, mut *"m`"> ->
def match_default_value(var m: MemType):
    match m^:
        case var y:
            _ = y.value


# A `ref` case binding refers to the SHARED storage.
# CHECK-LABEL: lit.fn @"match_ref_binding
# CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemType, mut *"s`", sugar_preserved({{.*}}, 2)>
# CHECK: %y = lit.var.decl "y" ref : !lit.ref<!lit.ref<!MemType, mut *"s`", sugar_preserved({{.*}}, 2)>
# CHECK: lit.ref.store %s, %y
def match_ref_binding():
    @__address_space(.SHARED)
    var s = MemType()
    match s:
        case ref y:
            _ = y.value


# Transferring the subject into a `var` binding reuses its SHARED storage, so
# nothing moves the value out of it.
# CHECK-LABEL: lit.fn @"match_transferred_var_binding
# CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemType, mut *"s`", sugar_preserved({{.*}}, 2)>
# CHECK-NEXT: lit.call {{.*}}MemType::@"__init__
# CHECK-NOT: lit.var.decl
# CHECK: lit.ref.struct.ger %s[value]
def match_transferred_var_binding():
    @__address_space(.SHARED)
    var s = MemType()
    match s^:
        case var y:
            _ = y.value


@fieldwise_init
struct Trivial(TrivialRegisterPassable):
    var x: Int


# A trivial copy out of SHARED is a plain load into a default address space var.
# CHECK-LABEL: lit.fn @"match_copies_trivial_value
# CHECK: %s = lit.var.decl "s" var : !lit.ref<!Trivial, mut *"s`", sugar_preserved({{.*}}, 2)>
# CHECK: %y = lit.var.decl "y" var : !lit.ref<!Trivial, mut *"y`{{[0-9]*}}">
# CHECK-NEXT: %[[VALUE:.*]] = lit.ref.load %s
# CHECK-NEXT: lit.ref.store %[[VALUE]], %y
def match_copies_trivial_value():
    @__address_space(.SHARED)
    var s = Trivial(1)
    match s:
        case var y:
            _ = y.x
