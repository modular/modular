# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @+1 {{function declared here}}
fn takes_pos_only_arg(a: Index, b: Index, /):
    pass


fn test_pos_only_arg_passed_by_kw(x: Index):
    # expected-error @+1 {{positional-only argument passed as keyword operand: 'b'}}
    takes_pos_only_arg(x, b=x)

    # expected-error @+1 {{positional-only arguments passed as keyword operands: 'a', 'b'}}
    takes_pos_only_arg(b=x, a=x)


# expected-note @+1 {{function declared here}}
fn takes_kw_only_arg(*, a: Index, b: Index, c: Index = `7`):
    pass


fn test_missing_kw_only_arg(x: Index):
    # COM: missing kw-only error takes precedence over unknown keyword
    # expected-error @+1 {{missing 1 required keyword-only argument: 'b'}}
    takes_kw_only_arg(a=x, d=x)

    # expected-error @+1 {{missing 2 required keyword-only arguments: 'a', 'b'}}
    takes_kw_only_arg()


# expected-note @+1 {{function declared here}}
fn takes_pos_or_kw_arg(i: Index, j: Index):
    pass


# expected-note @+1 {{function declared here}}
fn var_arg_func(*args: Index):
    pass


# expected-note @+1 {{declared here}}
fn pack_func[*Ts: AnyType](*args: *Ts):
    pass


fn test_unknown_kw_arg(x: Index):
    # expected-error @+1 {{unknown keyword argument: 'c'}}
    takes_pos_or_kw_arg(x, c=x, j=x)
    # expected-error @+1 {{unknown keyword arguments: 'd', 'c'}}
    takes_pos_or_kw_arg(x, d=x, c=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    var_arg_func(args=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    pack_func(args=x)


fn test_passed_by_pos_and_kw_arg(x: Index):
    # expected-error @+1 {{argument passed both as positional and keyword operand: 'i'}}
    takes_pos_or_kw_arg(x, i=x)

    # expected-error @+1 {{arguments passed both as positional and keyword operand: 'i', 'j'}}
    takes_pos_or_kw_arg(x, x, j=x, i=x)


# expected-note @+1 {{declared here}}
fn takes_pos_or_kw_param[i: Index, j: Index]():
    pass


fn test_unknown_kw_param[x: Index]():
    # expected-error @+1 {{unknown keyword parameter: 'c'}}
    takes_pos_or_kw_param[x, c=x, j=x]
    # expected-error @+1 {{unknown keyword parameters: 'd', 'c'}}
    takes_pos_or_kw_param[x, d=x, c=x]
    # expected-error @below {{unknown keyword parameter: 'Ts'}}
    pack_func[Ts=Index]


# expected-note @+1 {{function declared here}}
fn takes_pos_only_param[a: Index, b: Index, /]():
    pass


fn test_pos_only_param_passed_by_kw[x: Index]():
    # expected-error @+1 {{positional-only parameter passed as keyword operand: 'b'}}
    takes_pos_only_param[x, b=x]()

    # expected-error @+1 {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    takes_pos_only_param[b=x, a=x]()


# expected-note @+1 {{function declared here}}
fn takes_kw_only_param[*, a: Index, b: Index, c: Index = `7`]():
    pass


fn test_missing_kw_only_param[x: Index]():
    # expected-error @+1 {{unknown keyword parameter: 'd'}}
    takes_kw_only_param[a=x, d=x]()

    # expected-error @+1 {{missing 2 required keyword-only parameters: 'a', 'b'}}
    takes_kw_only_param[]()


# expected-note @below {{declared here}}
fn missing_keyword_only_params_tricky[a: Index, /, *, b: Index, c: Index = `3`]():
    pass


fn test_missing_keyword_only_params_tricky[x: Index]():
    # expected-error @below {{expects 1 positional parameter, but 3 were specified}}
    missing_keyword_only_params_tricky[x, x, x]


# expected-note @+1 {{function declared here}}
fn takes_kw_only_args(a: Index, b: Index, *args: Index, c: Index, d: Index = `2`):
    pass


fn test_missing_positional_arg_with_vararg_keyword(x: Index):
    # expected-error @+1 {{missing 1 required positional argument: 'b'}}
    takes_kw_only_args(x, c=`2`)


fn test_missing_keyword_arg_with_vararg_keyword(x: Index):
    takes_kw_only_args(x, x, c=`2`)


struct MemExample:
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


fn mutateMem(mut a: MemExample):
    pass
fn mutateInt(mut a: Int):
    pass

fn initialize_in_addrspace(memptr: UnsafePointer[MemExample, address_space=AddressSpace(1)],
                           regptr: UnsafePointer[Int, address_space=AddressSpace(1)]):
    # expected-error @+1 {{value of type 'MemExample' cannot be copied or moved into a non-default address space}}
    memptr[] = MemExample()
    # ok
    regptr[] = Int()


fn mutate_in_addrspace(memptr: UnsafePointer[MemExample, address_space=AddressSpace(1)],
                       regptr: UnsafePointer[Int, address_space=AddressSpace(1)]):
    # expected-error @+1 {{non-trivial value cannot be copied from a non-default address space}}
    mutateMem(memptr[])
    # ok
    mutateInt(regptr[])

fn variadic_addr_space(memptr: UnsafePointer[MemExample, address_space=AddressSpace(1)],
                       regptr: UnsafePointer[Int, address_space=AddressSpace(1)]):
    # expected-error @below {{non-trivial value cannot be copied from a non-default address space}}
    pack_func(memptr[])
    # Ok.
    pack_func(regptr[])


struct ParametricMutability:
    fn take_inout(mut self):  # expected-note {{function declared here}}
        # This is ok
        self.take_parametric()

    fn take_parametric(ref self):
        # expected-error @+1 {{invalid call to 'take_inout': invalid use of mutating method on rvalue of type 'ParametricMutability'}}
        self.take_inout()


fn test_ref[
    mut: Bool, origin: Origin[mut]._mlir_type
](ref [origin]arg: String):
    pass


fn call_test_ref(mut s: String):
    # expected-error @+1 {{cannot use parameterized function of type 'fn[Bool, Origin[$0.value]](ref [$1] arg: String) -> None' without binding all its parameters}}
    var f1 = test_ref

    # expected-error @+1 {{cannot use parameterized function of type 'fn[MutableOrigin](ref [$0] arg: String) -> None' without binding all its parameters}}
    var f2 = test_ref[True]
    f2(s)


@value
struct MyMutSpan[origin: MutableOrigin]:
    pass


fn take_two_spans(a: MyMutSpan[_], b: MyMutSpan[_]):
    # This is totally fine, can take two different mutable spans.
    pass


@value
struct MyStruct:
    var a: Int
    var b: Int


fn exclusivity[
    spanlife: MutableOrigin
](mut x: MyStruct, span: MyMutSpan[spanlife]):
    # Compiler injects a temporary to make this ok.
    x = x

    # Compiler injects a temporary to make this ok.
    x = x^

    # expected-error @below {{argument of 'take_two_spans' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'spanlife' memory accessed through reference embedded in value of type 'MyMutSpan[spanlife]'}}
    take_two_spans(span, span)


fn mutate_two[A: AnyType, B: AnyType](mut a: A, mut b: B):
    pass


fn mutate_two_AnyLifetime(
    ref [MutableAnyOrigin]a: Int, ref [MutableAnyOrigin]b: Int
):
    pass

fn mutate_variadic_any[T: AnyType](mut *values: T):
    pass

fn mutate_pack[*Ts: AnyType](mut *strs: *Ts):
    pass

fn inout_ref_exclusivity(mut a: Int, mut b: Int, mut s: MyStruct):
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
    # expected-note @below {{'s' memory accessed through reference embedded in value of type 'Int'}}
    mutate_two(s, s.a)

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


fn capture_exclusivity(owned x: MemExample):
    @parameter
    fn capture_and_read(y: MemExample):
        _ = x^

    # expected-error @below {{argument of call allows writing a memory location previously writable through implicit closure captures}}
    # expected-note @below {{'x' value is passed through aliasing 'read' argument}}
    capture_and_read(x)


# expected-note @below {{function declared here}}
fn param_inference_unrelated_error[T: AnyType](x: T, y: FloatLiteral[_]):
    pass


fn call_param_inference_unrelated_error():
    alias x = "hello"
    alias y = "world"
    # expected-error @below {{invalid call to 'param_inference_unrelated_error': failed to infer implicit parameter 'value' of argument 'y' type 'FloatLiteral'}}
    # expected-note @below {{failed to infer parameter #1, parameter isn't used in any argument}}
    param_inference_unrelated_error(x, y)


@value
@register_passable
struct MyRPStruct:
    var a: Int

    fn __del__(owned self):
        pass


@value
@register_passable
struct MyRPStruct2:
    var b: MyRPStruct

    fn __del__(owned self):
        pass


fn take_owned_and_mutate_rp(owned a: MyRPStruct2, mut b: MyRPStruct2):
    pass
fn rp_exclusivity(mut x: MyRPStruct2):
    # expected-error @below {{argument of 'take_owned_and_mutate_rp' call allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'mut' argument}}
    take_owned_and_mutate_rp(x^, x)


fn take_and_mutate_rp(a: MyRPStruct, mut b: MyRPStruct2):
    pass
fn rp_exclusivity2(mut x: MyRPStruct2):
    # expected-error @below {{argument of 'take_and_mutate_rp' call allows writing a memory location previously readable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'mut' argument}}
    take_and_mutate_rp(x.b, x)



# MOCO-1242 - [QoI] Improve error message on trait failure for variadics (e.g. print with Formattable)

# expected-note @below {{function declared here}}
fn my_print_variadic[*Ts: MyWritable](x: Int, *args: *Ts): pass
# expected-note @below {{function declared here}}
fn my_print_single[T: MyWritable](value: T): pass

fn test_print_errors(s: MyStruct):
  # expected-error @below {{invalid call to 'my_print_variadic': could not deduce parameter 'Ts' of callee 'my_print_variadic'}}
  # expected-note @below {{failed to infer parameter 'Ts', argument type 'MyStruct' does not conform to trait 'MyWritable'}}
  my_print_variadic(1, s)

  # expected-error @below {{invalid call to 'my_print_single': could not deduce parameter 'T' of callee 'my_print_single'}}
  # expected-note @below {{failed to infer parameter 'T', argument type 'MyStruct' does not conform to trait 'MyWritable'}}
  my_print_single(s)

trait MyWritable:
  fn method(self):
     pass
