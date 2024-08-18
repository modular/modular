# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @+1 {{function declared here}}
fn takes_pos_only_arg(a: int, b: int, /):
    pass


fn test_pos_only_arg_passed_by_kw(x: int):
    # expected-error @+1 {{positional-only argument passed as keyword operand: 'b'}}
    takes_pos_only_arg(x, b=x)

    # expected-error @+1 {{positional-only arguments passed as keyword operands: 'a', 'b'}}
    takes_pos_only_arg(b=x, a=x)


# expected-note @+1 {{function declared here}}
fn takes_kw_only_arg(*, a: int, b: int, c: int = `7`):
    pass


fn test_missing_kw_only_arg(x: int):
    # COM: missing kw-only error takes precedence over unknown keyword
    # expected-error @+1 {{missing 1 required keyword-only argument: 'b'}}
    takes_kw_only_arg(a=x, d=x)

    # expected-error @+1 {{missing 2 required keyword-only arguments: 'a', 'b'}}
    takes_kw_only_arg()


# expected-note @+1 {{function declared here}}
fn takes_pos_or_kw_arg(i: int, j: int):
    pass


# expected-note @+1 {{function declared here}}
fn var_arg_func(*args: int):
    pass


# expected-note @+1 {{declared here}}
fn pack_func[*Ts: AnyType](*args: *Ts):
    pass


fn test_unknown_kw_arg(x: int):
    # expected-error @+1 {{unknown keyword argument: 'c'}}
    takes_pos_or_kw_arg(x, c=x, j=x)
    # expected-error @+1 {{unknown keyword arguments: 'd', 'c'}}
    takes_pos_or_kw_arg(x, d=x, c=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    var_arg_func(args=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    pack_func(args=x)


fn test_passed_by_pos_and_kw_arg(x: int):
    # expected-error @+1 {{argument passed both as positional and keyword operand: 'i'}}
    takes_pos_or_kw_arg(x, i=x)

    # expected-error @+1 {{arguments passed both as positional and keyword operand: 'i', 'j'}}
    takes_pos_or_kw_arg(x, x, j=x, i=x)


# expected-note @+1 {{declared here}}
fn takes_pos_or_kw_param[i: int, j: int]():
    pass


fn test_unknown_kw_param[x: int]():
    # expected-error @+1 {{unknown keyword parameter: 'c'}}
    takes_pos_or_kw_param[x, c=x, j=x]
    # expected-error @+1 {{unknown keyword parameters: 'd', 'c'}}
    takes_pos_or_kw_param[x, d=x, c=x]
    # expected-error @below {{unknown keyword parameter: 'Ts'}}
    pack_func[Ts=int]


# expected-note @+1 {{function declared here}}
fn takes_pos_only_param[a: int, b: int, /]():
    pass


fn test_pos_only_param_passed_by_kw[x: int]():
    # expected-error @+1 {{positional-only parameter passed as keyword operand: 'b'}}
    takes_pos_only_param[x, b=x]()

    # expected-error @+1 {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    takes_pos_only_param[b=x, a=x]()


# expected-note @+1 {{function declared here}}
fn takes_kw_only_param[*, a: int, b: int, c: int = `7`]():
    pass


fn test_missing_kw_only_param[x: int]():
    # expected-error @+1 {{unknown keyword parameter: 'd'}}
    takes_kw_only_param[a=x, d=x]()

    # expected-error @+1 {{missing 2 required keyword-only parameters: 'a', 'b'}}
    takes_kw_only_param[]()


# expected-note @below {{declared here}}
fn missing_keyword_only_params_tricky[
    a: int, /, *, b: int, c: int
]():
    pass


fn test_missing_keyword_only_params_tricky[x: int]():
    # expected-error @below {{missing 2 required keyword-only parameters: 'b', 'c}}
    missing_keyword_only_params_tricky[x, x, x]


# expected-note @+1 {{function declared here}}
fn takes_kw_only_args(a: int, b: int, *args: int, c: int, d: int = `2`):
    pass


fn test_missing_positional_arg_with_vararg_keyword(x: int):
    # expected-error @+1 {{missing 1 required positional argument: 'b'}}
    takes_kw_only_args(x, c=`2`)


fn test_missing_keyword_arg_with_vararg_keyword(x: int):
    takes_kw_only_args(x, x, c=`2`)


struct MemExample:
  fn __init__(inout self): pass
  fn __copyinit__(inout self, existing: Self): pass

fn mutateMem(inout a: MemExample): pass

fn initialize_in_addrspace(ptr: UnsafePointer[MemExample, AddressSpace(1)]):
    # expected-error @+1 {{value of type 'MemExample' cannot be copied into a non-default address space}}
    ptr[] = MemExample()

fn mutate_in_addrspace(ptr: UnsafePointer[MemExample, AddressSpace(1)]):
    # expected-error @+1 {{value cannot be passed from a non-default address space}}
    mutateMem(ptr[])

struct ParametricMutability:
    fn take_inout(inout self): # expected-note {{function declared here}}
       # This is ok
       self.take_parametric()

    fn take_parametric(ref [_]self: Self):
        # expected-error @+1 {{invalid call to 'take_inout': invalid use of mutating method on rvalue of type 'ParametricMutability'}}
        self.take_inout()


fn test_ref[
    is_mutable: Bool, lifetime: AnyLifetime[is_mutable].type
](ref[lifetime] arg: String): pass


fn call_test_ref(inout s: String):
    # expected-error @+1 {{cannot use parameterized function of type 'fn[Bool, AnyLifetime[$0.value]](ref [$1] arg: String) -> None' without binding all its parameters}}
    var f1 = test_ref

    # expected-error @+1 {{cannot use parameterized function of type 'fn[MutableLifetime](ref [$0] arg: String) -> None' without binding all its parameters}}
    var f2 = test_ref[True]
    f2(s)


@value
struct MyMutSpan[
   lifetime: MutableLifetime
]: pass

fn take_two_spans(a: MyMutSpan[_], b: MyMutSpan[_]):
    # This is totally fine, can take two different mutable spans.
    pass


fn exclusivity[spanlife: MutableLifetime](inout x: MemExample, span: MyMutSpan[spanlife]):
    # expected-warning @below {{implicit __copyinit__ call argument allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'borrowed' argument}}
    x = x

    # TODO: This is not correctly diagnosed, due to transfer creating a novel
    # uncorrelated lifetime in the same space (!).
    x = x^

    # expected-warning @below {{call argument allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'x' value is passed through aliasing 'borrowed' argument}}
    x.__copyinit__(x)

    # expected-warning @below {{call argument allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'spanlife' memory accessed through reference embedded in value of type 'MyMutSpan[spanlife]'}}
    take_two_spans(span, span)

fn take_two_ints(inout a: Int, inout b: Int): pass

fn inout_ref_exclusivity(inout a: Int, inout b: Int):
    # This is ok.
    take_two_ints(a, b)

    # This is not.
    # expected-warning @below {{call argument allows writing a memory location previously writable through another aliased argument}}
    # expected-note @below {{'a' value is passed through aliasing 'inout' argument}}
    take_two_ints(a, a)


