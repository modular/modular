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


# expected-note @+1 {{function declared here}}
def takes_pos_only_arg(a: Int, b: Int, /):
    pass


def test_pos_only_arg_passed_by_kw(x: Int):
    # expected-error @+1 {{invalid call to 'takes_pos_only_arg': missing required argument: 'b'}}
    takes_pos_only_arg(x, b=x)

    # expected-error @+1 {{invalid call to 'takes_pos_only_arg': missing required argument: 'a'}}
    takes_pos_only_arg(b=x, a=x)


# expected-note @+1 {{function declared here}}
def takes_kw_only_arg(*, a: Int, b: Int, c: Int = 7):
    pass


def test_missing_kw_only_arg(x: Int):
    # COM: missing kw-only error takes precedence over unknown keyword
    # expected-error @+1 {{invalid call to 'takes_kw_only_arg': missing required argument: 'b'}}
    takes_kw_only_arg(a=x, d=x)

    # expected-error @+1 {{invalid call to 'takes_kw_only_arg': missing required argument: 'a'}}
    takes_kw_only_arg()


# expected-note @+1 {{function declared here}}
def takes_pos_or_kw_arg(i: Int, j: Int):
    pass


# expected-note @+1 {{function declared here}}
def var_arg_func(*args: Int):
    pass


# expected-note @+1 {{declared here}}
def pack_func[*Ts: AnyType](*args: *Ts):
    pass


def test_unknown_kw_arg(x: Int):
    # expected-error @+1 {{invalid call to 'takes_pos_or_kw_arg': unexpected keyword argument 'c'}}
    takes_pos_or_kw_arg(x, c=x, j=x)
    # expected-error @+1 {{invalid call to 'takes_pos_or_kw_arg': missing required argument: 'j'}}
    takes_pos_or_kw_arg(x, d=x, c=x)
    # expected-error @+1 {{invalid call to 'var_arg_func': unexpected keyword argument 'args'}}
    var_arg_func(args=x)
    # expected-error @+1 {{invalid call to 'pack_func': unexpected keyword argument 'args'}}
    pack_func(args=x)


def test_passed_by_pos_and_kw_arg(x: Int):
    # expected-error @+1 {{invalid call to 'takes_pos_or_kw_arg': missing required argument: 'j'}}
    takes_pos_or_kw_arg(x, i=x)

    # expected-error @+1 {{invalid call to 'takes_pos_or_kw_arg': unexpected keyword argument 'j'}}
    takes_pos_or_kw_arg(x, x, j=x, i=x)


# expected-note @+1 {{declared here}}
def takes_pos_or_kw_param[i: Int, j: Int]():
    pass


def test_unknown_kw_param[x: Int]():
    # expected-error @+1 {{unexpected keyword parameter 'c'}}
    takes_pos_or_kw_param[x, c=x, j=x]
    # expected-error @+1 {{unexpected keyword parameter 'd'}}
    takes_pos_or_kw_param[x, d=x, c=x]
    # expected-error @below {{unexpected keyword parameter 'Ts'}}
    pack_func[Ts=Int]


# expected-note @+1 {{function declared here}}
def takes_pos_only_param[a: Int, b: Int, /]():
    pass


def test_pos_only_param_passed_by_kw[x: Int]():
    # expected-error @+1 {{invalid call to 'takes_pos_only_param': unexpected keyword parameter 'b'}}
    takes_pos_only_param[x, b=x]()

    # expected-error @+1 {{invalid call to 'takes_pos_only_param': unexpected keyword parameter 'b'}}
    takes_pos_only_param[b=x, a=x]()


# expected-note @+1 {{function declared here}}
def takes_kw_only_param[*, a: Int, b: Int, c: Int = 7]():
    pass


def test_missing_kw_only_param[x: Int]():
    # expected-error @+1 {{invalid call to 'takes_kw_only_param': unexpected keyword parameter 'd'}}
    takes_kw_only_param[a=x, d=x]()

    # expected-error @+1 {{missing required keyword-only parameter: 'a'}}
    takes_kw_only_param[]()


# expected-note @below {{declared here}}
def missing_keyword_only_params_tricky[a: Int, /, *, b: Int, c: Int = 3]():
    pass


def test_missing_keyword_only_params_tricky[x: Int]():
    # expected-error @below {{unexpected parameter}}
    missing_keyword_only_params_tricky[x, x, x]


# expected-note @+1 {{function declared here}}
def takes_kw_only_args(a: Int, b: Int, *args: Int, c: Int, d: Int = 2):
    pass


def test_missing_positional_arg_with_vararg_keyword(x: Int):
    # expected-error @+1 {{invalid call to 'takes_kw_only_args': missing required argument: 'b'}}
    takes_kw_only_args(x, c=2)


def test_missing_keyword_arg_with_vararg_keyword(x: Int):
    takes_kw_only_args(x, x, c=2)


struct MemExample(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


struct MemExampleTriviallyCopyable(ImplicitlyCopyable):
    def __init__(out self):
        pass


def mutateMem(mut a: MemExample):
    pass


def mutateMemTC(mut a: MemExampleTriviallyCopyable):
    pass


def mutateInt(mut a: Int):
    pass


struct MemExampleMoveOnly(Movable):
    def __init__(out self):
        pass


def mutateMoveOnly(mut a: MemExampleMoveOnly):
    pass


def initialize_in_addrspace(
    memptr: Pointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)],
    regptr: Pointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)],
):
    # expected-error @+1 {{value of type 'MemExample' cannot be copied or moved into a non-default address space}}
    memptr[] = MemExample()
    # ok
    regptr[] = Int()


def mutate_in_addrspace(
    memptr: Pointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)],
    memtcptr: Pointer[
        MemExampleTriviallyCopyable,
        AnyOrigin[mut=True],
        address_space=AddressSpace(1)],
    regptr: Pointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)],
    moveptr: Pointer[
        MemExampleMoveOnly,
        AnyOrigin[mut=True],
        address_space=AddressSpace(1)],
):
    # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    mutateMem(memptr[])
    # expected-error @+2 {{value of type 'MemExampleMoveOnly' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @+1 {{consider transferring the value with '^'}}
    mutateMoveOnly(moveptr[])
    # ok
    mutateMemTC(memtcptr[])
    # ok
    mutateInt(regptr[])


def variadic_addr_space(
    memptr: Pointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)],
    regptr: Pointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)],
):
    # expected-error @below {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    pack_func(memptr[])
    # Ok.
    pack_func(regptr[])


struct ParametricMutability(Movable where False):
    def take_inout(mut self):  # expected-note {{function declared here}}
        # This is ok
        self.take_parametric()

    def take_parametric(ref self):
        # expected-error @+1 {{invalid call to 'take_inout': invalid use of mutating method on rvalue of type 'ParametricMutability'}}
        self.take_inout()


def test_ref[mut: Bool, //, origin: Origin[mut=mut]](ref[origin] arg: String):
    pass


def call_test_ref(mut s: String):
    # expected-error @below {{cannot use parametric function as a runtime closure}}
    # expected-note @below {{parameter 'mut' of type 'Bool' is not bound}}
    var f1 = test_ref

    # This is ok, because the only unbound parameter is the origin which is a
    # singleton type.
    var f2 = test_ref[mut=True, ...]
    # Can now bind the remaining parameter in a call.
    f2(s)


@fieldwise_init
struct MyStruct(ImplicitlyCopyable):
    var a: Int
    var b: Int


# expected-note @+1 {{function declared here}}
def mutate_pack[*Ts: AnyType](mut *strs: *Ts):
    pass


# expected-note @+1 {{function declared here}}
def consume_owned_variadic_pack[*Ts: AnyType](var *inner: *Ts):
    pass


def forward_borrowed_pack_to_mut_pack[*Ts: AnyType](*outer: *Ts):
    # expected-error @below {{cannot unpack a variadic pack into a call that requires a stricter mutability}}
    mutate_pack(*outer)


def forward_unknown_mut_pack_to_mut_pack[
    *Ts: AnyType
](outer: VariadicPack[origin=_, element_trait=AnyType, _, *Ts]):
    # expected-error @below {{cannot unpack a variadic pack into a call that requires a different ownership}}
    mutate_pack(*outer)


def forward_borrowed_pack_to_owned_pack[*Ts: AnyType](*outer: *Ts):
    # expected-error @below {{cannot unpack a variadic pack into a call that requires a different ownership}}
    consume_owned_variadic_pack(*outer)


def forward_unknown_ownership_pack[
    *Ts: AnyType, owned: Bool
](outer: VariadicPack[origin=_, element_trait=AnyType, owned, *Ts]):
    # expected-error @below {{cannot unpack a variadic pack into a call that requires a different ownership}}
    consume_owned_variadic_pack(*outer)


# expected-note @below {{function declared here}}
def param_inference_unrelated_error[T: AnyType](x: T, y: FloatLiteral[_]):
    pass


def call_param_inference_unrelated_error():
    comptime x = "hello"
    comptime y = "world"
    # expected-error @below {{value passed to 'y' cannot be converted from 'StringLiteral["world"]' to 'FloatLiteral[y.value]', it depends on an unresolved parameter 'y.value'}}
    param_inference_unrelated_error(x, y)


# MOCO-1242 - [QoI] Improve error message on trait failure for variadics (e.g. print with Formattable)
# expected-note @below {{function declared here}}
def my_print_variadic[*Ts: MyWritable](x: Int, *args: *Ts):
    pass


# expected-note @below {{function declared here}}
def my_print_single[T: MyWritable](value: T):
    pass


def test_print_errors(s: MyStruct):
    # expected-error @below {{invalid call to 'my_print_variadic': an element of 'args' with type 'MyStruct' does not conform to trait 'MyWritable'; either prove the conformance with 'conforms_to', or add conformance}}
    my_print_variadic(1, s)

    # expected-error @below {{invalid call to 'my_print_single': value passed to 'value' cannot be converted from 'MyStruct' to 'T', argument type 'MyStruct' does not conform to trait 'MyWritable'}}
    my_print_single(s)


trait MyWritable:
    def method(self):
        pass


# Issue #4499: https://github.com/modular/modular/issues/4499
# Traits with ref self cause issues when used as parameter
trait MyTrait4499:
    def method(ref self):
        ...


struct MyStruct4499(MyTrait4499, Movable where False):
    def method(ref self):
        pass


struct Owner4499[T: MyTrait4499](Movable where False):
    def __init__(out self):
        pass


def my_func4499(arg0: Owner4499, arg1: Owner4499):
    pass


def test_4499_exclusivity():
    # Should be ok.
    my_func4499(Owner4499[MyStruct4499](), Owner4499[MyStruct4499]())


# Test printing of apply expressions.
def vararg_example(*args: Int, other: Int):
    pass


def pack_example[*Ts: AnyType](*args: *Ts, other: Int):
    pass


def generic_example[T: AnyType, //](a: T):
    pass


@fieldwise_init
struct StructWithFlexParam[T: AnyType, //, x: T](Movable where False):
    pass


# expected-note @+1 {{function declared here}}
def takeWith4(a: StructWithFlexParam[4]):
    pass


def test_print_apply_expressions():
    # expected-note @below {{.T of the first type is '__MLIRType[None]' but the second type is 'Int'}}
    # expected-error @below {{invalid call to 'takeWith4': value passed to 'a' cannot be converted from 'StructWithFlexParam[vararg_example(Int(0), Int(1), Int(2), Int(3), Int(4), other=Int(5))]' to 'StructWithFlexParam[Int(4)]'}}
    takeWith4(StructWithFlexParam[vararg_example(0, 1, 2, 3, 4, other=5)]())
    # expected-note @below {{.T of the first type is '__MLIRType[None]' but the second type is 'Int'}}
    # expected-error @below {{invalid call to 'takeWith4': value passed to 'a' cannot be converted from 'StructWithFlexParam[pack_example[Int, String](Int(0), String("foo"), other=Int(5))]' to 'StructWithFlexParam[Int(4)]'}}
    takeWith4(StructWithFlexParam[pack_example(0, "foo", other=5)]())
    # expected-note @below {{.T of the first type is '__MLIRType[None]' but the second type is 'Int'}}
    # expected-error @below {{from 'StructWithFlexParam[generic_example(Int(0))]' to}}
    takeWith4(StructWithFlexParam[generic_example(0)]())
    # expected-note @below {{.T of the first type is '__MLIRType[None]' but the second type is 'Int'}}
    # expected-error @below {{invalid call to 'takeWith4': value passed to 'a' cannot be converted from 'StructWithFlexParam[vararg_example(other=Int(5))]' to 'StructWithFlexParam[Int(4)]'}}
    takeWith4(StructWithFlexParam[vararg_example(other=5)]())


struct DifferExample[shape: Int, addr: Int](Movable where False):
    pass


def differ_take[  # expected-note {{function declared here}}
    tile_impl: def[input_addr: Int, output_shape: Int, output_addr: Int](
        DifferExample[output_shape, output_addr],
    ) thin -> None,
]():
    pass


def differ_wrong[
    input_addr: Int, output_shape: Int, output_addr: Int
](output: DifferExample[output_shape, input_addr]):
    # expected-error @below {{has 'def[input_addr: Int, output_shape: Int, output_addr: Int](DifferExample[output_shape, output_addr]) thin -> None' type, but value has type 'def differ_wrong[input_addr: Int, output_shape: Int, output_addr: Int](output: DifferExample[output_shape, input_addr]) thin -> None'}}
    # expected-note @below {{.output.addr of the first value is 'output_addr' but the second value is 'input_addr'}}
    differ_take[differ_wrong]()


# expected-note @below {{cannot be converted from 'StringLiteral[""]' to 'Int'}}
def call_error_location(a: Int):
    pass


# expected-note @below {{cannot be converted from 'StringLiteral[""]' to 'IntLiteral}}
def call_error_location(a: IntLiteral):
    pass


def test_call_error_location():
    # The error location should be on the operand, not the call.
    call_error_location(
        # expected-error @below {{no matching function in call to 'call_error_location'}}
        ""
    )
