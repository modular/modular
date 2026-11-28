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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


@fieldwise_init
struct MyStruct(ImplicitlyCopyable):
    var a: Int
    var b: Int


def mutate_two[A: AnyType, B: AnyType](mut a: A, mut b: B):
    pass


def take_two_owned[A: AnyType, B: AnyType](var a: A, var b: B):
    pass


def mutate_one_read_one[A: AnyType, B: AnyType](mut a: A, b: B):
    pass


def read_one_mutate_one[A: AnyType, B: AnyType](a: A, mut b: B):
    pass


def read_two[A: AnyType, B: AnyType](a: A, b: B):
    pass


# ===----------------------------------------------------------------------=== #
# Basic, field-sensitive, and variadic exclusivity.
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct MyMutSpan[origin: Origin[mut=True]](Movable where False):
    pass


def take_two_spans(a: MyMutSpan[_], b: MyMutSpan[_]):
    pass


def test_nested_origin_exclusivity[spanlife: MutOrigin](
    span: MyMutSpan[spanlife],
):
    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'take_two_spans' call}}
    # expected-note @below {{'origin_of(spanlife)' memory accessed through reference embedded in value of type 'MyMutSpan[spanlife]'}}
    take_two_spans(span, span)


def mutate_two_any_origin(
    ref[AnyOrigin[mut=True]] a: Int,
    ref[AnyOrigin[mut=True]] b: Int,
):
    pass


def mutate_variadic_any[T: AnyType](mut *values: T):
    pass


def mutate_pack[*Ts: AnyType](mut *values: *Ts):
    pass


def test_argument_exclusivity(mut a: Int, mut b: Int, mut s: MyStruct):
    mutate_two(a, b)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'mut' argument}}
    mutate_two(a, a)

    mutate_two(s.a, s.b)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(s.a)' value is passed through aliasing 'mut' argument}}
    mutate_two(s.a, s.a)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(s)' value is passed through aliasing 'mut' argument}}
    mutate_two(s.a, s)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(s.a)' value is passed through aliasing 'mut' argument}}
    mutate_two(s, s.a)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'take_two_owned' call}}
    # expected-note @below {{'origin_of(s)' value is passed through aliasing 'var' argument}}
    take_two_owned(s^, s^)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(s.a)' value is passed through aliasing 'imm' argument}}
    mutate_one_read_one(s, s.a)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two_any_origin' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'ref' argument}}
    mutate_two_any_origin(a, a)

    mutate_variadic_any[Int]()
    mutate_variadic_any(s)
    mutate_variadic_any(a, b)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_variadic_any' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(a, a)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_variadic_any' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(a, b, a)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_variadic_any' call}}
    # expected-note @below {{'origin_of(s)' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(s, s)

    mutate_pack(a)
    mutate_pack(a, s)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_pack' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'mut' argument}}
    mutate_pack(a, a)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_pack' call}}
    # expected-note @below {{'origin_of(a)' value is passed through aliasing 'mut' argument}}
    mutate_pack(a, b, a)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_pack' call}}
    # expected-note @below {{'origin_of(s)' value is passed through aliasing 'mut' argument}}
    mutate_pack(s, s)

def exclusivity[
    spanlife: Origin[mut=True]
](mut x: MyStruct, span: MyMutSpan[spanlife]):
    # Compiler injects a temporary to make this ok.
    x = x

    # Compiler injects a temporary to make this ok.
    x = x^

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'take_two_spans' call}}
    # expected-note @below {{'origin_of(spanlife)' memory accessed through reference embedded in value of type 'MyMutSpan[spanlife]'}}
    take_two_spans(span, span)


# ===----------------------------------------------------------------------=== #
# Register-passable and result-slot exclusivity.
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct MyRPStruct(RegisterPassable):
    var a: Int

    def __deinit__(deinit self):
        pass


@fieldwise_init
struct MyRPStruct2(RegisterPassable):
    var b: MyRPStruct

    def __deinit__(deinit self):
        pass


def take_owned_and_mutate_rp(var a: MyRPStruct2, mut b: MyRPStruct2):
    pass


def take_and_mutate_rp(a: MyRPStruct, mut b: MyRPStruct2):
    pass


def test_register_passable_exclusivity(mut value: MyRPStruct2):
    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'take_owned_and_mutate_rp' call}}
    # expected-note @below {{'origin_of(value)' value is passed through aliasing 'mut' argument}}
    take_owned_and_mutate_rp(value^, value)

    # expected-error @below {{aliasing values passed immutably to 'a' argument and passed mutably to 'b' argument in 'take_and_mutate_rp' call}}
    # expected-note @below {{'origin_of(value)' value is passed through aliasing 'mut' argument}}
    take_and_mutate_rp(value.b, value)


struct Buf(Movable where False):
    var data: List[Int]

    def view(self) -> ref[self.data[0]] Int:
        return self.data[0]

    def __init__(out self, ref size: Int):
        self.data = {}


def test_result_slot_exclusivity():
    var value: Buf
    # expected-error @below {{aliasing values passed immutably to 'size' argument and constructed as a result in 'Buf' initializer call}}
    # expected-note @below {{introduce a temporary to avoid mutating the call result while accessing it through an argument}}
    value = Buf(value.view())


# ===----------------------------------------------------------------------=== #
# Nested, subtree, and interior origin exclusivity.
# ===----------------------------------------------------------------------=== #

@__unsafe_nested_origins_read_only
def nested_mutability_disabled1(ref a: String, b: String):
    pass


@__unsafe_nested_origins_read_only
def nested_mutability_disabled2(mut a: String, b: String):
    pass


def test_nested_mutability_disabled():
    var value: String
    nested_mutability_disabled1(value, value)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'nested_mutability_disabled2' call}}
    # expected-note @below {{'origin_of(value)' value is passed through aliasing 'imm' argument}}
    nested_mutability_disabled2(value, value)


@__unsafe_nested_origins_read_only
def get_subtree_ref[T: AnyType, //, subtree_root: Origin](
    ref[subtree_root.subtree] value: T,
) -> ref[subtree_root.subtree] T:
    return value


def test_subtree_origin_exclusivity(
    mut value: MyStruct,
    mut lhs: Int,
    mut rhs: Int,
):
    mutate_two(value.a, value.b)
    mutate_two(
        get_subtree_ref[origin_of(lhs)](lhs),
        get_subtree_ref[origin_of(rhs)](rhs),
    )
    mutate_two(get_subtree_ref[origin_of(value.a)](value.a), value.b)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(value.b)' value is passed through aliasing 'mut' argument 'b'}}
    mutate_two(get_subtree_ref[origin_of(value)](value.a), value.b)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(origin_of(value).subtree)' value is passed through aliasing 'mut' argument 'b'}}
    mutate_two(
        get_subtree_ref[origin_of(value)](value.a),
        get_subtree_ref[origin_of(value)](value.b),
    )

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed mutably to 'b' argument in 'mutate_two' call}}
    # expected-note @below {{'origin_of(value.a)' value is passed through aliasing 'mut' argument 'b'}}
    mutate_two(get_subtree_ref[origin_of(value.a)](value.a), value.a)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(value.a)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(
        get_subtree_ref[origin_of(value)](value.a),
        value.a,
    )


struct InteriorOwner:
    var data: List[Int]
    var other: Int

    def first(
        ref self,
    ) -> ref[origin_of(self)._get_owned_interior["first"]] Int:
        abort()

    def second(
        ref self,
    ) -> ref[origin_of(self)._get_owned_interior["second"]] Int:
        abort()


def test_interior_origin_exclusivity(
    mut values: List[Int],
    mut nested_values: List[List[Int]],
    mut owner: InteriorOwner,
):
    read_two(values, values[0])
    mutate_two(owner.data[0], owner.other)
    mutate_two(owner.first(), owner.second())

    # expected-error @below {{aliasing values passed immutably to 'a' argument and passed mutably to 'b' argument in 'read_one_mutate_one' call}}
    # expected-note @below {{'origin_of(values["element"])' value is passed through aliasing 'mut' argument 'b'}}
    read_one_mutate_one(values, values[0])

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(values)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(values[0], values)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(values["element"])' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(values[0], values[1])

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(nested_values["element"])' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(nested_values[0][0], nested_values[0])

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(nested_values)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(nested_values[0][0], nested_values)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(owner.data)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(owner.data[0], owner.data)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(owner)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(owner.data[0], owner)

    # expected-error @below {{aliasing values passed mutably to 'a' argument and passed immutably to 'b' argument in 'mutate_one_read_one' call}}
    # expected-note @below {{'origin_of(origin_of(values).subtree)' value is passed through aliasing 'imm' argument 'b'}}
    mutate_one_read_one(
        values[0],
        get_subtree_ref[origin_of(values)](values),
    )

    # expected-error @below {{aliasing values passed immutably to 'a' argument and passed mutably to 'b' argument in 'read_one_mutate_one' call}}
    # expected-note @below {{'origin_of(values["element"])' value is passed through aliasing 'mut' argument 'b'}}
    read_one_mutate_one(
        get_subtree_ref[origin_of(values)](values),
        values[0],
    )


# ===----------------------------------------------------------------------=== #
# Variadic pack exclusivity.
# ===----------------------------------------------------------------------=== #

def read_pack[*Ts: AnyType](*values: *Ts):
    pass


def mutate_pack_values[*Ts: AnyType](mut *values: *Ts):
    pass


def test_mut_pack_aliasing[*Ts: AnyType](mut *values: *Ts):
    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_pack_values' call}}
    # expected-note @below {{'origin_of(values.origin)' value is passed through aliasing 'mut' argument 'values' by a '*' unpack}}
    mutate_pack_values(*values, *values)

    # expected-error @below {{aliasing values passed mutably to 'values' argument and passed mutably to 'values' argument in 'mutate_pack_values' call}}
    # expected-note @below {{'origin_of(values.origin)' value is passed through aliasing 'mut' argument 'values' by a '*' unpack}}
    mutate_pack_values(values[0], *values)

    read_pack(*values, *values)
    read_pack(values[0], *values)
    var local = 0
    mutate_pack_values(local, *values)


# ===----------------------------------------------------------------------=== #
# Closure capture exclusivity.
# ===----------------------------------------------------------------------=== #

struct CaptureList[T: AnyType](Movable where False):
    var data: Pointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = Pointer[
            Self.T, UntrackedOrigin[mut=True]
        ].unsafe_dangling()

    def __deinit__(deinit self):
        pass

    def __getitem__(
        ref self,
    ) -> ref[self.data._get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
        return self.data._get_ref_with_unsafe_interior_origin["element"](self)


def test_capture_container_and_mutable_element[origin: MutOrigin](
    ref[origin] values: CaptureList[Int],
):
    var pointer = Pointer(to=values[])

    @always_inline
    def inner() {imm}:
        _ = values
        _ = pointer[]

    # expected-error @below {{aliasing values passed immutably to '' argument and passed mutably to '' argument in '__call__' call}}
    # expected-note @below {{'origin_of(origin["element"])' memory accessed through reference embedded in value of type 'def() -> None'}}
    inner()


# ===----------------------------------------------------------------------=== #
# Aggregate construction exclusivity.
# ===----------------------------------------------------------------------=== #

def test_array_pointer_exclusivity():
    var lhs = 1
    var rhs = 2

    # FIXME: This should work after exclusivity checking understands that each
    # loop iteration accesses only one element of the union.
    # expected-error @below {{aliasing values passed mutably to 'elements' argument and passed mutably to 'elements' argument in 'Array[Pointer[Int, origin_of(lhs, rhs)], Int(2)]' initializer call}}
    # expected-note @below {{'origin_of(lhs)' memory accessed through reference embedded in value of type 'Pointer[Int, origin_of(lhs, rhs)]'}}
    # expected-note @below {{'origin_of(rhs)' memory accessed through reference embedded in value of type 'Pointer[Int, origin_of(lhs, rhs)]'}}
    for element in [Pointer(to=lhs), Pointer(to=rhs)]:
        element[] *= 2

# ===----------------------------------------------------------------------=== #
# "accesses" exclusivity.
# ===----------------------------------------------------------------------=== #


def no_pointee_access[origin: MutOrigin](
    lhs: Pointer[Int, origin],
    rhs: Pointer[Int, origin],
    mut value: Int,
) __accesses(none origin):
    pass


def no_pointee_access_with_aliasing_args[origin: MutOrigin](
    ptr: Pointer[Int, origin],
    mut lhs: Int,
    mut rhs: Int,
) __accesses(none origin):
    pass


def immutable_pointer_storage[origin: MutOrigin](
    mut ptr: Pointer[Int, origin],
    mut value: Int,
) __accesses(imm ptr):
    pass


def read_first_field(
    mut value: MyStruct,
    mut other: Int,
) __accesses(imm value.a):
    pass


def test_refined_exclusivity():
    var value = 0
    no_pointee_access(Pointer(to=value), Pointer(to=value), value)

    # Refining the pointee origin does not refine either argument's storage.
    # expected-error @below {{aliasing values passed mutably to 'lhs' argument and passed mutably to 'rhs' argument in 'no_pointee_access_with_aliasing_args' call}}
    # expected-note @below {{'origin_of(value)' value is passed through aliasing 'mut' argument 'rhs'}}
    no_pointee_access_with_aliasing_args(Pointer(to=value), value, value)

    var ptr = Pointer(to=value)
    # Access to the argument's storage is independent of its pointee origin.
    # expected-error @below {{aliasing values passed mutably to 'ptr' argument and passed mutably to 'value' argument in 'immutable_pointer_storage' call}}
    # expected-note @below {{'origin_of(value)' value is passed through aliasing 'mut' argument 'value'}}
    immutable_pointer_storage(ptr, value)

    var fields = MyStruct(1, 2)
    read_first_field(fields, fields.b)

    # expected-error @below {{aliasing values passed mutably to 'other' argument and passed immutably to 'value' argument in 'read_first_field' call}}
    # expected-note @below {{'origin_of(fields.a)' memory accessed through reference embedded in value of type 'MyStruct'}}
    read_first_field(fields, fields.a)
