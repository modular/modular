# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @+1 {{function declared here}}
def takes_pos_only_arg(a: Int, b: Int, /):
    pass


def test_pos_only_arg_passed_by_kw(x: Int):
    # expected-error @+1 {{positional-only argument passed as keyword operand: 'b'}}
    takes_pos_only_arg(x, b=x)

    # expected-error @+1 {{positional-only arguments passed as keyword operands: 'a', 'b'}}
    takes_pos_only_arg(b=x, a=x)


# expected-note @+1 {{function declared here}}
def takes_kw_only_arg(*, a: Int, b: Int, c: Int = 7):
    pass


def test_missing_kw_only_arg(x: Int):
    # COM: missing kw-only error takes precedence over unknown keyword
    # expected-error @+1 {{missing 1 required keyword-only argument: 'b'}}
    takes_kw_only_arg(a=x, d=x)

    # expected-error @+1 {{missing 2 required keyword-only arguments: 'a', 'b'}}
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
    # expected-error @+1 {{unknown keyword argument: 'c'}}
    takes_pos_or_kw_arg(x, c=x, j=x)
    # expected-error @+1 {{unknown keyword arguments: 'd', 'c'}}
    takes_pos_or_kw_arg(x, d=x, c=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    var_arg_func(args=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    pack_func(args=x)


def test_passed_by_pos_and_kw_arg(x: Int):
    # expected-error @+1 {{argument passed both as positional and keyword operand: 'i'}}
    takes_pos_or_kw_arg(x, i=x)

    # expected-error @+1 {{arguments passed both as positional and keyword operand: 'i', 'j'}}
    takes_pos_or_kw_arg(x, x, j=x, i=x)


# expected-note @+1 {{declared here}}
def takes_pos_or_kw_param[i: Int, j: Int]():
    pass


def test_unknown_kw_param[x: Int]():
    # expected-error @+1 {{unknown keyword parameter: 'c'}}
    takes_pos_or_kw_param[x, c=x, j=x]
    # expected-error @+1 {{unknown keyword parameters: 'd', 'c'}}
    takes_pos_or_kw_param[x, d=x, c=x]
    # expected-error @below {{unknown keyword parameter: 'Ts'}}
    pack_func[Ts=Int]


# expected-note @+1 {{function declared here}}
def takes_pos_only_param[a: Int, b: Int, /]():
    pass


def test_pos_only_param_passed_by_kw[x: Int]():
    # expected-error @+1 {{positional-only parameter passed as keyword operand: 'b'}}
    takes_pos_only_param[x, b=x]()

    # expected-error @+1 {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    takes_pos_only_param[b=x, a=x]()


# expected-note @+1 {{function declared here}}
def takes_kw_only_param[*, a: Int, b: Int, c: Int = 7]():
    pass


def test_missing_kw_only_param[x: Int]():
    # expected-error @+1 {{unknown keyword parameter: 'd'}}
    takes_kw_only_param[a=x, d=x]()

    # expected-error @+1 {{missing 2 required keyword-only parameters: 'a', 'b'}}
    takes_kw_only_param[]()


# expected-note @below {{declared here}}
def missing_keyword_only_params_tricky[a: Int, /, *, b: Int, c: Int = 3]():
    pass


def test_missing_keyword_only_params_tricky[x: Int]():
    # expected-error @below {{'missing_keyword_only_params_tricky' expects 1 positional parameter, but 3 were specified}}
    missing_keyword_only_params_tricky[x, x, x]


# expected-note @+1 {{function declared here}}
def takes_kw_only_args(a: Int, b: Int, *args: Int, c: Int, d: Int = 2):
    pass


def test_missing_positional_arg_with_vararg_keyword(x: Int):
    # expected-error @+1 {{missing 1 required positional argument: 'b'}}
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


def initialize_in_addrspace(
    memptr: UnsafePointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
    regptr: UnsafePointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
):
    # expected-error @+1 {{value of type 'MemExample' cannot be copied or moved into a non-default address space}}
    memptr[] = MemExample()
    # ok
    regptr[] = Int()


def mutate_in_addrspace(
    memptr: UnsafePointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
    memtcptr: UnsafePointer[
        MemExampleTriviallyCopyable,
        AnyOrigin[mut=True],
        address_space=AddressSpace(1),
    ],
    regptr: UnsafePointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
):
    # expected-error @+1 {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    mutateMem(memptr[])
    # ok
    mutateMemTC(memtcptr[])
    # ok
    mutateInt(regptr[])


def variadic_addr_space(
    memptr: UnsafePointer[
        MemExample, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
    regptr: UnsafePointer[
        Int, AnyOrigin[mut=True], address_space=AddressSpace(1)
    ],
):
    # expected-error @below {{non-implicitly trivially copyable value cannot be copied from a non-default address space}}
    pack_func(memptr[])
    # Ok.
    pack_func(regptr[])


struct ParametricMutability:
    def take_inout(mut self):  # expected-note {{function declared here}}
        # This is ok
        self.take_parametric()

    def take_parametric(ref self):
        # expected-error @+1 {{invalid call to 'take_inout': invalid use of mutating method on rvalue of type 'ParametricMutability'}}
        self.take_inout()


def test_ref[mut: Bool, //, origin: Origin[mut=mut]](ref[origin] arg: String):
    pass


def call_test_ref(mut s: String):
    # expected-error @+1 {{cannot use parameterized function of type 'def[mut: Bool, _, +, origin: Origin[mut=mut]](ref[_mlir_origin] arg: String) -> None' without binding all its parameters}}
    var f1 = test_ref

    # expected-error @+1 {{cannot use parameterized function of type 'def[_, +, origin: MutOrigin](ref[_mlir_origin] arg: String) -> None' without binding all its parameters}}
    var f2 = test_ref[mut=True, ...]
    # expected-error @+1 {{cannot call dynamic function with parameterized type}}
    f2(s)


@fieldwise_init
struct MyMutSpan[origin: Origin[mut=True]]:
    pass


def take_two_spans(a: MyMutSpan[_], b: MyMutSpan[_]):
    # This is totally fine, can take two different mutable spans.
    pass


@fieldwise_init
struct MyStruct(ImplicitlyCopyable):
    var a: Int
    var b: Int


def exclusivity[
    spanlife: Origin[mut=True]
](mut x: MyStruct, span: MyMutSpan[spanlife]):
    # Compiler injects a temporary to make this ok.
    x = x

    # Compiler injects a temporary to make this ok.
    x = x^

    # expected-error @below {{argument of 'take_two_spans' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'spanlife' memory accessed through reference embedded in value of type 'MyMutSpan[spanlife]'}}
    take_two_spans(span, span)


def mutate_two[A: AnyType, B: AnyType](mut a: A, mut b: B):
    pass


def take_two_owned[A: AnyType, B: AnyType](var a: A, var b: B):
    pass


def mutate_one_read_one[A: AnyType, B: AnyType](mut a: A, b: B):
    pass


def mutate_two_AnyLifetime(
    ref[AnyOrigin[mut=True]] a: Int, ref[AnyOrigin[mut=True]] b: Int
):
    pass


def mutate_variadic_any[T: AnyType](mut *values: T):
    pass


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


def inout_ref_exclusivity(mut a: Int, mut b: Int, mut s: MyStruct):
    # This is ok.
    mutate_two(a, b)

    # This is not.
    # expected-error @below {{argument of 'mutate_two' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'mut' argument}}
    mutate_two(a, a)

    # This is ok: field sensitivity.
    mutate_two(s.a, s.b)

    # expected-error @below {{argument of 'mutate_two' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s.a' value is passed through aliasing 'mut' argument}}
    mutate_two(s.a, s.a)

    # expected-error @below {{argument of 'mutate_two' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s' value is passed through aliasing 'mut' argument}}
    mutate_two(s.a, s)

    # expected-error @below {{argument of 'mutate_two' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s.a' value is passed through aliasing 'mut' argument}}
    mutate_two(s, s.a)

    # expected-error @below {{argument of 'take_two_owned' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s' value is passed through aliasing 'var' argument}}
    take_two_owned(s^, s^)

    # expected-error @below {{argument of 'mutate_one_read_one' call allows reading a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s.a' value is passed through aliasing 'read' argument}}
    mutate_one_read_one(s, s.a)

    # expected-error @below {{argument of 'mutate_two_AnyLifetime' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'ref' argument}}
    mutate_two_AnyLifetime(a, a)

    # These are all ok.
    mutate_variadic_any[Int]()
    mutate_variadic_any(s)
    mutate_variadic_any(a, b)

    # expected-error @below {{argument of 'mutate_variadic_any' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(a, a)

    # expected-error @below {{argument of 'mutate_variadic_any' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(a, b, a)

    # expected-error @below {{argument of 'mutate_variadic_any' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s' value is passed through aliasing 'mut' argument}}
    mutate_variadic_any(s, s)

    # These are ok.
    mutate_pack(a)
    mutate_pack(a, s)

    # expected-error @below {{argument of 'mutate_pack' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'mut' argument}}
    mutate_pack(a, a)

    # expected-error @below {{argument of 'mutate_pack' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'mut' argument}}
    mutate_pack(a, b, a)

    # expected-error @below {{argument of 'mutate_pack' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'s' value is passed through aliasing 'mut' argument}}
    mutate_pack(s, s)


def capture_exclusivity(var x: MemExample):
    @parameter
    def capture_and_read(y: MemExample):
        _ = x^

    # FIXME(MOCO-3241): Re-enable this.
    # xpected-error @below {{argument of call allows reading a memory location previously writable through implicit closure captures}}
    # xpected-note @below {{'x' value is passed through aliasing 'read' argument}}
    capture_and_read(x)


# expected-note @below {{function declared here}}
def param_inference_unrelated_error[T: AnyType](x: T, y: FloatLiteral[_]):
    pass


def call_param_inference_unrelated_error():
    comptime x = "hello"
    comptime y = "world"
    # expected-error @below {{value passed to 'y' cannot be converted from 'StringLiteral["world"]' to 'FloatLiteral[y.value]', it depends on an unresolved parameter 'y.value'}}
    param_inference_unrelated_error(x, y)


@fieldwise_init
struct MyRPStruct(RegisterPassable):
    var a: Int

    def __del__(deinit self):
        pass


@fieldwise_init
struct MyRPStruct2(RegisterPassable):
    var b: MyRPStruct

    def __del__(deinit self):
        pass


def take_owned_and_mutate_rp(var a: MyRPStruct2, mut b: MyRPStruct2):
    pass


def rp_exclusivity(mut x: MyRPStruct2):
    # expected-error @below {{argument of 'take_owned_and_mutate_rp' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'mut' argument}}
    take_owned_and_mutate_rp(x^, x)


def take_and_mutate_rp(a: MyRPStruct, mut b: MyRPStruct2):
    pass


def rp_exclusivity2(mut x: MyRPStruct2):
    # expected-error @below {{argument of 'take_and_mutate_rp' call allows writing a memory location previously readable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'mut' argument}}
    take_and_mutate_rp(x.b, x)


# MOCO-1242 - [QoI] Improve error message on trait failure for variadics (e.g. print with Formattable)


# expected-note @below {{function declared here}}
def my_print_variadic[*Ts: MyWritable](x: Int, *args: *Ts):
    pass


# expected-note @below {{function declared here}}
def my_print_single[T: MyWritable](value: T):
    pass


def test_print_errors(s: MyStruct):
    # expected-error @below {{invalid call to 'my_print_variadic': could not convert element of 'args' with type 'MyStruct' to expected type 'MyWritable'}}
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


struct MyStruct4499(MyTrait4499):
    def method(ref self):
        pass


struct Owner4499[T: MyTrait4499]:
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
struct StructWithFlexParam[T: AnyType, //, x: T]:
    pass


# expected-note @+1 {{function declared here}}
def takeWith4(a: StructWithFlexParam[4]):
    pass


def test_print_apply_expressions():
    # expected-note @below {{.T of left type is '__MLIRType[None]' but the right type is 'Int'}}
    # expected-error @below {{from 'StructWithFlexParam[vararg_example(0, 1, 2, 3, 4, other=5)]' to}}
    takeWith4(StructWithFlexParam[vararg_example(0, 1, 2, 3, 4, other=5)]())
    # expected-note @below {{.T of left type is '__MLIRType[None]' but the right type is 'Int'}}
    # expected-error @below {{from 'StructWithFlexParam[pack_example[Int, String](0, String("foo"), other=5)]' to}}
    takeWith4(StructWithFlexParam[pack_example(0, "foo", other=5)]())
    # expected-note @below {{.T of left type is '__MLIRType[None]' but the right type is 'Int'}}
    # expected-error @below {{from 'StructWithFlexParam[generic_example(0)]' to}}
    takeWith4(StructWithFlexParam[generic_example(0)]())
    # expected-note @below {{.T of left type is '__MLIRType[None]' but the right type is 'Int'}}
    # expected-error @below {{from 'StructWithFlexParam[vararg_example(other=5)]' to}}
    takeWith4(StructWithFlexParam[vararg_example(other=5)]())
