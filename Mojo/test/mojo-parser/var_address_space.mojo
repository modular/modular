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


struct Storage(Movable):
    var value: Int

    def __init__[
        addr_space: AddressSpace
    ](out[addr_space] self, *, uninitialized: Bool):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))


# A pointer to the var gets the var's origin and address space, and a store
# through it writes to the var.
# CHECK-LABEL: lit.fn @"pointer_to_var
# CHECK: %p = lit.var.decl "p" var : !lit.ref<!lit.struct<#Pointer <{{.*}} *"x`", {{.*}}, :!AddressSpace {_value: !SIMDLength = {_mlir_value = sugar_preserved({{.*}}, 2)}}>>, mut *"p`{{[0-9]*}}">
# CHECK: %[[REF:.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}} -> !lit.ref<!Storage, mut *"x`", sugar_preserved({{.*}}, 2)>>
# CHECK: lit.ref.struct.ger %[[REF]][value] : <!Storage, mut *"x`", sugar_preserved({{.*}}, 2)>
def pointer_to_var():
    @__address_space(.SHARED)
    var x = Storage(uninitialized=True)
    var p = Pointer(to=x)
    p[].value = 1


# CHECK-LABEL: lit.fn @"pointer_to_parametric_var
# CHECK: %p = lit.var.decl "p" var : !lit.ref<!lit.struct<#Pointer <{{.*}} *"x`", {{.*}}, :!AddressSpace addr_space>>, mut *"p`{{[0-9]*}}">
# CHECK: %[[REF:.*]] = lit.call {{.*}}Pointer::@"__getitem__{{.*}} -> !lit.ref<!Storage, mut *"x`", #lit.struct.extract<{{.*}}addr_space, "_value">, "_mlir_value">>>
# CHECK: lit.ref.struct.ger %[[REF]][value] : <!Storage, mut *"x`", #lit.struct.extract<{{.*}}addr_space, "_value">, "_mlir_value">>
def pointer_to_parametric_var[addr_space: AddressSpace]():
    @__address_space(addr_space)
    var x = Storage(uninitialized=True)
    var p = Pointer(to=x)
    p[].value = 1


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


struct Overloaded(ImplicitlyCopyable):
    var value: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

    def get(self) -> Int:
        return 0

    def get[AS: AddressSpace](ref[_, AS] self) -> Int:
        return 1

    def bump(mut self):
        pass

    def bump[AS: AddressSpace](ref[_, AS] self):
        pass

    # The same pair in the other declaration order.
    def bump_ref_first[AS: AddressSpace](ref[_, AS] self):
        pass

    def bump_ref_first(mut self):
        pass


# An 'imm self' or 'mut self' overload copies a value in SHARED into the
# default address space, so the overload that takes 'self' by reference in its
# own address space is selected.
# CHECK-LABEL: lit.fn @"overload_on_address_space
# CHECK-NOT: lit.memcpy
# CHECK: lit.call @{{.*}}Overloaded::@"get[::AddressSpace,
# CHECK-NOT: lit.memcpy
# CHECK: lit.call @{{.*}}Overloaded::@"bump[::AddressSpace,
# CHECK-NOT: lit.memcpy
# CHECK: lit.call @{{.*}}Overloaded::@"bump_ref_first[::AddressSpace,
# CHECK-NOT: lit.memcpy
# CHECK: lit.return
def overload_on_address_space():
    @__address_space(.SHARED)
    var o = Overloaded()
    _ = o.get()
    o.bump()
    o.bump_ref_first()


# In the default address space, 'imm self' and 'mut self' bind fewer parameters
# and win.
# CHECK-LABEL: lit.fn @"overload_in_default
# CHECK: lit.call @{{.*}}Overloaded::@"get(
# CHECK: lit.call @{{.*}}Overloaded::@"bump(
# CHECK: lit.call @{{.*}}Overloaded::@"bump_ref_first(
# CHECK: lit.return
def overload_in_default():
    var o = Overloaded()
    _ = o.get()
    o.bump()
    o.bump_ref_first()


@fieldwise_init
struct RegOverloaded(TrivialRegisterPassable):
    var value: Int

    def get(self) -> Int:
        return 0

    def get[AS: AddressSpace](ref[_, AS] self) -> Int:
        return 1

    def bump(mut self):
        pass

    def bump[AS: AddressSpace](ref[_, AS] self):
        pass


# A register passable value in SHARED also selects the 'ref' overloads. A
# 'mut self' overload would write a copy of the value back after the call.
# CHECK-LABEL: lit.fn @"register_passable_overload_on_address_space
# CHECK: [[INIT:%.*]] = lit.call {{.*}}RegOverloaded::@"__init__(
# CHECK-NEXT: [[R:%.*]] = lit.var.decl "r" var : !lit.ref<!RegOverloaded, mut *"r`", {{.*}}, 2)>
# CHECK-NEXT: lit.ref.store [[INIT]], [[R]]
# CHECK-NEXT: [[GET:%.*]] = lit.call {{.*}}RegOverloaded::@"get[::AddressSpace,{{.*}}([[R]])
# CHECK-NEXT: lit.ownership.use [[GET]]
# CHECK-NEXT: lit.call {{.*}}RegOverloaded::@"bump[::AddressSpace,{{.*}}([[R]])
# CHECK: lit.return
def register_passable_overload_on_address_space():
    @__address_space(.SHARED)
    var r = RegOverloaded(1)
    _ = r.get()
    r.bump()


# CHECK-LABEL: lit.fn @"register_passable_overload_in_default
# CHECK: [[INIT:%.*]] = lit.call {{.*}}RegOverloaded::@"__init__(
# CHECK: [[R:%.*]] = lit.var.decl "r" var : !lit.ref<!RegOverloaded, mut *"r`">
# CHECK: lit.ref.store [[INIT]], [[R]] : <!RegOverloaded, mut *"r`">
# CHECK: [[VAL:%.*]] = lit.ref.load [[R]] : <!RegOverloaded, mut *"r`">
# CHECK: lit.call {{.*}}RegOverloaded::@"get({{.*}}([[VAL]])
# CHECK: lit.call {{.*}}RegOverloaded::@"bump({{.*}}([[R]])
# CHECK: lit.return
def register_passable_overload_in_default():
    var r = RegOverloaded(1)
    _ = r.get()
    r.bump()


def take_trivial(t: Trivial) -> Int:
    return 0


def take_trivial(var t: Trivial) -> Int:
    return 1


# Both an 'imm' and a 'var' argument copy the value out of SHARED, so their
# order is the same as in the default address space.
# CHECK-LABEL: lit.fn @"imm_and_var_overloads_in_shared
# CHECK: lit.call {{.*}}@"take_trivial(var_address_space::Trivial)"
# CHECK: lit.return
def imm_and_var_overloads_in_shared():
    @__address_space(.SHARED)
    var s = Trivial(1)
    _ = take_trivial(s)


struct CrossAddressSpaceCopy(ImplicitlyCopyable):
    var x: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.x = 0

    def __init__(out self, *, copy: Self):
        self.x = copy.x

    def __init__[
        from_as: AddressSpace, to_as: AddressSpace
    ](out[to_as] self, *, ref[from_as] copy: Self):
        self.x = copy.x


def marker(): pass

# An implicit copy of a value in SHARED selects the copy initializer that takes
# the value by reference in its own address space.
# CHECK-LABEL: lit.fn @"implicit_copy_from_shared

# CHECK: [[S:%.*]] = lit.var.decl "s" var : !lit.ref<!CrossAddressSpaceCopy, mut *"s`", {{.*}}, 2)>
# CHECK: lit.call @{{.*}}CrossAddressSpaceCopy::@"__init__[::AddressSpace,{{.*}}([[S]])
# CHECK: lit.call tail @var_address_space::@"marker()"()
# CHECK: [[Y:%.*]] = lit.var.decl "y" var : !lit.ref<!CrossAddressSpaceCopy, mut *"y`1">
# CHECK: lit.call @{{.*}}CrossAddressSpaceCopy::@"__init__[::AddressSpace,::AddressSpace,{{.*}}copy{{.*}}([[S]], [[Y]])
# CHECK: lit.call tail @var_address_space::@"marker()"()
# CHECK: [[Z:%.*]] = lit.var.decl "z" var : !lit.ref<!CrossAddressSpaceCopy, mut *"z`2">
# CHECK: lit.call @{{.*}}CrossAddressSpaceCopy::@"__init__[::AddressSpace,::AddressSpace,{{.*}}copy{{.*}}([[S]], [[Z]])
# CHECK: lit.return
def implicit_copy_from_shared():
    @__address_space(.SHARED)
    var s = CrossAddressSpaceCopy()
    marker()
    match s:
        case var y:
            _ = y.x
    marker()
    var z = s
    marker()
    _ = z.x


# A value in a parametric address space also selects the 'ref' overload.
# CHECK-LABEL: lit.fn @"overload_on_parametric_address_space
# CHECK: lit.call @{{.*}}Overloaded::@"get[::AddressSpace,
# CHECK: lit.return
def overload_on_parametric_address_space[addr_space: AddressSpace]():
    @__address_space(addr_space)
    var o = Overloaded()
    _ = o.get()


struct GlobalOnly(TrivialRegisterPassable):
    var value: Int

    def __init__[addr_space: AddressSpace](out[addr_space] self):
        self.value = 0

    def get(self) -> Int:
        return 0

    def get[
        AS: AddressSpace
    ](ref[_, AS] self) -> Int where AS == AddressSpace.GLOBAL:
        return 1


# The where clause is false for SHARED, so only the 'imm self' overload is
# viable.
# CHECK-LABEL: lit.fn @"where_clause_excludes_ref_overload
# CHECK: lit.call {{.*}}GlobalOnly::@"get(
# CHECK: lit.return
def where_clause_excludes_ref_overload():
    @__address_space(.SHARED)
    var g = GlobalOnly()
    _ = g.get()


# CHECK-LABEL: lit.fn @"where_clause_allows_ref_overload
# CHECK: lit.call @{{.*}}GlobalOnly::@"get[::AddressSpace,
# CHECK: lit.return
def where_clause_allows_ref_overload():
    @__address_space(.GLOBAL)
    var g = GlobalOnly()
    _ = g.get()


# CHECK-LABEL: lit.fn @"where_clause_with_evidence
# CHECK: lit.call @{{.*}}GlobalOnly::@"get[::AddressSpace,
# CHECK: lit.return
def where_clause_with_evidence[
    addr_space: AddressSpace
]() where addr_space == AddressSpace.GLOBAL:
    @__address_space(addr_space)
    var g = GlobalOnly()
    _ = g.get()
