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

def identity(x: Int) -> Int:
    return x


# expected-error @below {{cannot implicitly convert 'def identity(x: Int) thin -> Int' value to 'def(Int) -> Int'}}
comptime invalid_bare_fn_type: def(Int) -> Int = identity


struct MemType(Movable where False):
    pass


def mut_ship_function(mut x: MemType):
    ...


# We can convert from def(imm MemType)->None to def(mut MemType)->None but not
# vice versa (see TTSMFS).
# expected-error @below {{cannot implicitly convert 'def mut_ship_function(mut x: MemType) thin -> None' value to 'def(MemType) thin -> None' in comptime initializer}}
comptime read_ship_fn_alias: def(imm MemType) thin -> None = mut_ship_function


def may_mutate_nested_origins(x: Int):
    pass


# expected-note @below {{function declared here}}
def requires_nested_origins_read_only(
    f: def(Int) thin __accesses("__unsafe_nested_origins_read_only"),
):
    pass


def test_nested_origins_read_only_conversion():
    # expected-error @below {{function may mutate nested origins}}
    requires_nested_origins_read_only(may_mutate_nested_origins)


@__unsafe_nested_origins_read_only
def legacy_nested_origins_read_only(x: Int):
    pass


@__nested_origins_read_only
def checked_nested_origins_read_only(x: Int):
    pass


# expected-note @below {{function declared here}}
def requires_checked_nested_origins_read_only(
    f: def(Int) thin __accesses("__nested_origins_read_only"),
):
    pass


def test_checked_nested_origins_read_only_conversion():
    # expected-error @below {{function may mutate nested origins}}
    requires_checked_nested_origins_read_only(may_mutate_nested_origins)
    requires_checked_nested_origins_read_only(legacy_nested_origins_read_only)
    requires_checked_nested_origins_read_only(checked_nested_origins_read_only)


struct AccessSubtypePoint:
    var x: Int
    var y: Int


def unrestricted_access(mut x: Int):
    pass


def read_access(mut x: Int) __accesses(imm x):
    pass


def read_whole_point(
    mut point: AccessSubtypePoint
) __accesses(imm point):
    pass


def mutate_whole_point(mut point: AccessSubtypePoint):
    pass


def mutate_x_read_y(
    mut point: AccessSubtypePoint
) __accesses(mut point.x, imm point.y):
    pass


# expected-note @below {{function declared here}}
def requires_read_access(
    f: def(mut x: Int) thin __accesses(imm x),
):
    pass


# expected-note @below {{function declared here}}
def requires_no_access(
    f: def(mut x: Int) thin __accesses(none x),
):
    pass


# expected-note @below {{function declared here}}
def requires_read_x(
    f: def(
        mut point: AccessSubtypePoint
    ) thin __accesses(imm point.x),
):
    pass


# expected-note @below {{function declared here}}
def requires_mut_x_read_y(
    f: def(
        mut point: AccessSubtypePoint
    ) thin __accesses(mut point.x, imm point.y),
):
    pass


# expected-note @below {{function declared here}}
def requires_read_x_mut_y(
    f: def(
        mut point: AccessSubtypePoint
    ) thin __accesses(imm point.x, mut point.y),
):
    pass


def test_refined_access_function_conversion_errors():
    # expected-error @below {{function may exceed the permitted accesses for argument 'x'}}
    requires_read_access(unrestricted_access)
    # expected-error @below {{function may exceed the permitted accesses for argument 'x'}}
    requires_no_access(read_access)
    # expected-error @below {{function may exceed the permitted accesses for argument 'point'}}
    requires_read_x(read_whole_point)
    # An unrestricted mutable argument cannot satisfy field-level promises.
    # expected-error @below {{function may exceed the permitted accesses for argument 'point'}}
    requires_mut_x_read_y(mutate_whole_point)
    # The fields are compared independently; mutable x cannot satisfy readonly x.
    # expected-error @below {{function may exceed the permitted accesses for argument 'point'}}
    requires_read_x_mut_y(mutate_x_read_y)


# A redundant refinement is equivalent to the argument's default contract.
# expected-warning @+3 {{access specification is redundant with the argument's default access}}
comptime redundant_expected_access: def(
    mut x: Int
) thin __accesses(mut x) = unrestricted_access


# A nested-origins decorator has no effect when the method's type has no
# nested origins to refine.
struct NoNestedOrigins:
    # expected-warning @below {{nested-origins read-only decorator has no effect because this method has no nested origins; remove the decorator}}
    @__nested_origins_read_only
    def example1(self):
        pass

    # expected-warning @below {{nested-origins read-only decorator has no effect because this method has no nested origins; remove the decorator}}
    @__unsafe_nested_origins_read_only
    def example2(self):
        pass

    # expected-warning @below {{nested-origins read-only decorator has no effect because this method has no nested origins; remove the decorator}}
    @__unsafe_nested_origins_read_only
    def example3[o: Origin](ref self, ref [o] arg: Int):
        pass

    # No warning, there is a nested origin in the pointer.
    @__unsafe_nested_origins_read_only
    def example4[o: Origin](ref self, arg: Pointer[Int, o]):
        pass

    # No warning, `T` may have trait methods that mutate nested origins.
    @__nested_origins_read_only
    def example5[T: AnyType](self, arg: T):
        pass


struct GenericNoNestedOrigins[T: AnyType]:
    # No warning, `T` may have trait methods that mutate nested origins.
    @__nested_origins_read_only
    def example(self, arg: Self.T):
        pass


# expected-note @below {{trait 'NestedOriginsReadOnlyRequirement' declared here}}
trait NestedOriginsReadOnlyRequirement:
    @__unsafe_nested_origins_read_only
    # expected-note @below {{no 'read' candidates have type 'def(self: MissingNestedOriginsReadOnly[origin]) thin -> Int __accesses("__unsafe_nested_origins_read_only")'}}
    def read(self) -> Int:
        ...


# expected-error @below {{'MissingNestedOriginsReadOnly[origin]' does not implement all requirements for 'NestedOriginsReadOnlyRequirement'}}
struct MissingNestedOriginsReadOnly[origin: Origin](
    NestedOriginsReadOnlyRequirement
):
    var ptr: Pointer[Int, Self.origin]

    # expected-note @below {{candidate declared here with type}}
    def read(self) -> Int:
        return 0


# expected-note @below {{trait 'CheckedNestedOriginsReadOnlyRequirement' declared here}}
trait CheckedNestedOriginsReadOnlyRequirement:
    @__nested_origins_read_only
    # expected-note @below {{no 'read' candidates have type 'def(self: MissingCheckedNestedOriginsReadOnly[origin]) thin -> Int __accesses("__nested_origins_read_only")'}}
    def read(self) -> Int:
        ...


# expected-error @below {{'MissingCheckedNestedOriginsReadOnly[origin]' does not implement all requirements for 'CheckedNestedOriginsReadOnlyRequirement'}}
struct MissingCheckedNestedOriginsReadOnly[origin: Origin](
    CheckedNestedOriginsReadOnlyRequirement
):
    var ptr: Pointer[Int, Self.origin]

    # expected-note @below {{candidate declared here with type}}
    def read(self) -> Int:
        return 0


# expected-error @below {{unsupported experimental access set "unknown"}}
def unsupported_access_set(x: Int) __accesses("unknown"):
    pass


# expected-warning @below {{access specification is redundant with the argument's default access}}
def redundant_access_spec(mut x: Int) __accesses(mut x):
    pass


# expected-error @below {{access specification may not expand the argument's default access}}
comptime ExpandedImmAccess = def(imm y: String) thin __accesses(mut y)


# expected-error @below {{access specification may not expand the argument's default access}}
comptime ExpandedParametricRefAccess = def[origin: Origin](ref[origin] y: String) thin __accesses(mut y)


# expected-error @below {{'var' arguments are always mutable and cannot have refined access specifications}}
def refined_var_access(var x: Int) __accesses(imm x):
    pass


# expected-error @below {{'out' arguments cannot have refined access specifications}}
def refined_out_access(out x: Int) __accesses(imm x):
    pass


# expected-error @below {{variadic arguments and packs cannot have refined access specifications}}
def refined_varargs_access(*xs: Int) __accesses(imm xs):
    pass


# expected-error @below {{variadic arguments and packs cannot have refined access specifications}}
def refined_pack_access[*Ts: AnyType](*xs: *Ts) __accesses(imm xs):
    pass


# expected-error @below {{'__accesses' requires at least one access specifier}}
def empty_access_set(x: Int) __accesses():
    pass


# expected-note @below {{previous access specification is here}}
# expected-error @below {{overlapping origin access specifications}}
def overlapping_accesses(mut x: Int) __accesses(imm x, mut x):
    pass


struct RefinedAccessErrorPoint:
    var x: Int


# expected-note @below {{previous access specification is here}}
# expected-error @below {{overlapping origin access specifications}}
def overlapping_field_accesses(mut point: RefinedAccessErrorPoint) __accesses(imm point, mut point.x):
    pass


# expected-error @below {{access origin must be a name or field path}}
def computed_access(mut x: Int) __accesses(imm String(x)):
    pass


# expected-error @below {{access specifications may not use AnyOrigin, origin unions, subtree origins, or interior origins}}
def any_origin_access(x: Int) __accesses(imm MutAnyOrigin):
    pass


# expected-error @+3 {{access specifications may not use AnyOrigin, origin unions, subtree origins, or interior origins}}
def subtree_origin_access[origin: Origin](
    ref[origin] x: Int
) __accesses(imm origin.subtree):
    pass


# expected-error @+3 {{access origin is reachable through multiple arguments; qualify it through an argument name}}
def ambiguous_explicit_origin_access[origin: Origin](
    ref[origin] x: Int, ref[origin] y: Int
) __accesses(imm origin):
    pass


# Tests that we detect too few arguments when doing function conversions.


# expected-note @below {{function declared here}}
def infer_variadic[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(x: Int, y: Int, * args: * ArgTypes) thin -> None,
]():
    pass


def device_func(i: Int):
    pass


def test_infer_variadic():
    # expected-error @below {{converted from 'def device_func(i: Int) thin -> None' to 'def(x: Int, y: Int, *args: *ArgTypes) thin -> None', expected at least 2 arguments, found 1}}
    infer_variadic[device_func]()


# ===----------------------------------------------------------------------=== #
# Tests that we correctly match each incoming argument type against the
# callee's variadic's element trait.
struct ZInt(Movable where False):
    pass


trait Sprongling:
    def sprongle(self):
        ...

# expected-note @below {{function declared here}}
def issue1_infer_variadic[
    ArgTypes: TypeList[Trait=Sprongling, ...],
    //,
    func: def(* args: * ArgTypes) thin -> None,
]():
    pass


def issue1_device_func(i: ZInt, j: ZInt):
    pass


def issue1_test_infer_variadic():
    # expected-error @below {{converted from 'def issue1_device_func(i: ZInt, j: ZInt) thin -> None' to 'def(*args: *ArgTypes) thin -> None', argument 'i' of type 'ZInt' does not match variadic element type 'Sprongling'}}
    issue1_infer_variadic[issue1_device_func]()


# ===----------------------------------------------------------------------=== #
def issue2_device_func(a: Int, b: Bool) -> Int:
    return 73


@fieldwise_init
struct issue2_DeviceFunction[*ArgTypes: TrivialRegisterPassable]:
    # expected-note @below {{function declared here}}
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


def issue2_compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> issue2_DeviceFunction[*ArgTypes]:
    return issue2_DeviceFunction[*ArgTypes]()


def issue2_main():
    var thing = issue2_compile[issue2_device_func]()
    # expected-error @below {{invalid call to 'call': value passed to 'args' cannot be converted from 'StringLiteral["hello"]' to 'Bool'}}
    var result2 = thing.call(42, "hello")


# ===----------------------------------------------------------------------=== #

# Tests that we reject any device_function that is still generic.


# A function that has a T that can't be inferred from anything
def issue3_device_func_unusedT[T: AnyType](a: Int, b: Bool) -> Int:
    return 73


# A function that has a T that can be inferred from arguments
def issue3_device_func_usedT[T: AnyType](a: T, b: Bool) -> Int:
    return 73


@fieldwise_init
struct issue3_DeviceFunction[*ArgTypes: TrivialRegisterPassable]:
    pass


# expected-note @below {{function declared here}}
def issue3_compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> issue3_DeviceFunction[*ArgTypes]:
    return issue3_DeviceFunction[*ArgTypes]()


def test_reject_generic_device_func_unusedT():
    # expected-error @below {{converted from 'def issue3_device_func_unusedT[T: AnyType](a: Int, b: Bool) thin -> Int' to 'def(*args: *ArgTypes) thin -> Int', compile-time parameter 'T' has no match in the expected type}}
    var thing = issue3_compile[issue3_device_func_unusedT]()


# Slightly different case, for no particular reason
def test_reject_generic_device_func_usedT():
    # expected-error @below {{converted from 'def issue3_device_func_usedT[T: AnyType](a: T, b: Bool) thin -> Int' to 'def(*args: *ArgTypes) thin -> Int', compile-time parameter 'T' has no match in the expected type}}
    var thing = issue3_compile[issue3_device_func_usedT]()


# ===----------------------------------------------------------------------=== #
def issue4_device_func(a: Int, b: Bool) -> Int:
    return 73


@fieldwise_init
struct issue4_DeviceFunction[*ArgTypes: TrivialRegisterPassable]:
    # expected-note @below {{function declared here}}
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


def issue4_compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> issue4_DeviceFunction[*ArgTypes]:
    return issue4_DeviceFunction[*ArgTypes]()


def issue4_main():
    var thing = issue4_compile[issue4_device_func]()
    # expected-error @below {{invalid call to 'call': value passed to 'args' cannot be converted from 'StringLiteral["hello"]' to 'Bool'}}
    var result2 = thing.call(42, "hello")


# ===----------------------------------------------------------------------=== #
# Tests that we can properly reject a `raises` function when handed to a
# non-raising input-parameter def.


def issue5_device_func(a: Int, b: Bool) raises -> Int:
    return 73


# expected-note @below {{function declared here}}
def issue5_compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]():
    pass


def issue5_main():
    # expected-error @below {{converted from 'def issue5_device_func(a: Int, b: Bool) raises thin -> Int' to 'def(*args: *ArgTypes) thin -> Int', function must not be 'raises'}}
    issue5_compile[issue5_device_func]()


# ===----------------------------------------------------------------------=== #
# Tests a GPU-function-like case (see FTAGPUF) but this one catches when the
# user hands in something of the wrong type.
# TODO(MOCO-1106): This doesn't catch when the given argument type
# (DeviceBuffer[Int]) mismatches the expected argument type
# (Pointer[Float32]) by only the input parameter (Int vs Float32), that's
# only caught by elaborator for now and is tested elsewhere (search FTAGPUF).

@fieldwise_init
struct ZBool(Movable where False):
    pass


trait ConvertibleToZPointer:
    comptime Pointee: AnyType

    def to_zpointer(self) -> ZPointer[Self.Pointee]:
        ...


struct ZPointer[T: AnyType](TrivialRegisterPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZPointer](out self, c: C):
        # TODO(MOCO-1106): If we can remove this rebind, we win. We'd need to
        # constrain C.Pointee=T somehow, or make ConvertibleToZPointer into a
        # generic trait instead of using an associated alias.
        # As it is, this won't catch incorrectly passing in a e.g.
        # ZDeviceBuffer[Int] into a ZPointer[Bool].
        var z: ZPointer[Self.T] = rebind[ZPointer[Self.T]](
            c.to_zpointer()
        )


trait ConvertibleToZTileTensor:
    def to_tensor(self) -> ZTileTensor:
        ...


struct ZTileTensor(TrivialRegisterPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__[C: ConvertibleToZTileTensor](out self, c: C):
        var z: ZTileTensor = c.to_tensor()


@fieldwise_init
struct issue6_DeviceFunction[*ArgTypes: TrivialRegisterPassable]:
    # expected-note @below {{function declared here}}
    def call(self, *args: * Self.ArgTypes) -> Int:
        return 91


@fieldwise_init
struct ManagedTileTensor(ConvertibleToZTileTensor, Movable where False):
    def to_tensor(self) -> ZTileTensor:
        return ZTileTensor()


# Never converted, the GPU just uses this one directly
@fieldwise_init
struct NDBuffer(TrivialRegisterPassable):
    pass


def kernel(t: ZTileTensor, p: ZPointer[Int], n: NDBuffer) -> Int:
    return 73


def issue6_compile[
    ArgTypes: TypeList[Trait=TrivialRegisterPassable, ...],
    //,
    func: def(* args: * ArgTypes) thin -> Int,
]() -> issue6_DeviceFunction[*ArgTypes]:
    return issue6_DeviceFunction[*ArgTypes]()


def issue6_main():
    var thing = issue6_compile[kernel]()
    var mtt = ManagedTileTensor()
    var ndb = NDBuffer()
    # This ZBool() is incorrect, not even close to the ZPointer[Int] that's
    # expected.
    # expected-error @below {{invalid call to 'call': value passed to 'args' cannot be converted from 'ZBool' to 'ZPointer[Int]'}}
    var result1 = thing.call(mtt, ZBool(), ndb)


# Tests that a plain def type and a abi("C") def type with the same signature
# are not interchangeable: plain → abi("C") is rejected.


# expected-note @below {{function declared here}}
def takes_abi_c(f: def(Int) thin abi("C") -> Int):
    pass


def plain(x: Int) -> Int:
    return x


def test_plain_to_abi_c():
    # expected-error @below {{cannot be converted from 'def plain(x: Int) thin -> Int' to 'def(Int) abi("C") thin -> Int', function must be 'abi("C")'}}
    takes_abi_c(plain)


# Tests that a plain def type and a abi("C") def type with the same signature
# are not interchangeable: abi("C") → plain is rejected.


# expected-note @below {{function declared here}}
def takes_plain(f: def(Int) thin -> Int):
    pass


def test_abi_c_to_plain(f: def(Int) thin abi("C") -> Int):
    # expected-error @below {{cannot be converted from 'def(Int) abi("C") thin -> Int' to 'def(Int) thin -> Int', function must not be 'abi("C")'}}
    takes_plain(f)


# Tests that defining a plain function and a abi("C") function with the same
# name and parameter types is a redefinition error: the calling-convention
# effect is not an overload discriminator.


# expected-note @below {{previous definition here}}
def redef_abi_mixed(x: Int) -> Int:
    return x


# expected-error @below {{redefinition of function 'redef_abi_mixed' with identical signature}}
def redef_abi_mixed(x: Int) abi("C") -> Int:
    return x


# Tests that a abi("C") nested function that captures variables is rejected.
# C ABI has no closure mechanism, so a capturing abi("C") function would
# silently corrupt argument registers.

def test_abi_c_capturing():
    var x: Int = 1

    # expected-error @below {{a abi("C") function cannot capture variables}}
    def captures_x(y: Int) abi("C") {imm} -> Int:
        return x + y

    _ = captures_x(2)


# Tests that specifying 'thin' twice on a function type is an error,
# even when combined with abi("C").


def test_duplicate_thin_with_abi_c():
    # expected-error @below {{function effect 'thin' was already specified; remove the duplicate}}
    var _: def(Int) thin thin abi("C") -> Int


# Function-type conversion where the expected callee's variadic element
# type is a metatype rather than a trait (MOCO-2210). Surfaces as a
# regular conversion error instead of crashing in `matchFunctionTypes`.


struct MyStruct(Movable where False):
    pass


# expected-note @below {{function declared here}}
def infer_meta_variadic[
    ArgTypes: TypeList[Trait=type_of(MyStruct), ...],
    //,
    func: def(* args: * ArgTypes) thin -> None,
]():
    pass


def plain_func(i: Int):
    pass


def test_infer_meta_variadic():
    # expected-error @below {{invalid call to 'infer_meta_variadic': value passed to 'func' cannot be converted from 'def plain_func(i: Int) thin -> None' to 'def(*args: *ArgTypes) thin -> None', argument 'i' of type 'Int' does not match variadic element type 'AnyStruct[MyStruct]'}}
    infer_meta_variadic[plain_func]()


# The following tests check the reason given for a function type mismatch.


# expected-note @below {{function declared here}}
def takes_two(f: def(Int, Int) thin -> None):
    pass


def one_arg(x: Int):
    pass


def test_arg_count():
    # expected-error @below {{cannot be converted from 'def one_arg(x: Int) thin -> None' to 'def(Int, Int) thin -> None', expected 2 arguments, found 1}}
    takes_two(one_arg)


# expected-note @below {{function declared here}}
def takes_read(f: def(Int) thin -> None):
    pass


def mut_arg(mut x: Int):
    pass


def test_convention():
    # expected-error @below {{cannot be converted from 'def mut_arg(mut x: Int) thin -> None' to 'def(Int) thin -> None', argument 'x': expected 'imm' convention, found 'mut'}}
    takes_read(mut_arg)


# expected-note @below {{function declared here}}
def takes_kw(f: def(*, a: Int) thin -> None):
    pass


def kw_b(*, b: Int):
    pass


def test_keyword_name():
    # expected-error @below {{cannot be converted from 'def kw_b(*, b: Int) thin -> None' to 'def(*, a: Int) thin -> None', expected keyword-only argument 'a', found 'b'}}
    takes_kw(kw_b)


# expected-note @below {{function declared here}}
def takes_no_kwargs(f: def(Int) thin -> None):
    pass


def with_kwargs(x: Int, var **kwargs: Int):
    pass


def test_kwargs():
    # expected-error @below {{cannot be converted from 'def with_kwargs(x: Int, var **kwargs: Int) thin -> None' to 'def(Int) thin -> None', function must not take a '**kwargs' argument}}
    takes_no_kwargs(with_kwargs)
