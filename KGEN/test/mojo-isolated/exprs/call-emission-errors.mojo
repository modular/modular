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
    # expected-error @+1 {{got 1 positional-only argument passed as keyword operand: 'b'}}
    takes_pos_only_arg(x, b=x)

    # expected-error @+1 {{got 2 positional-only arguments passed as keyword operands: 'a', 'b'}}
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
fn pack_func[*Ts: AnyRegType](*args: *Ts):
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
    # expected-error @+1 {{argument #0 ('i') passed both as positional and keyword operand}}
    takes_pos_or_kw_arg(x, i=x)


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
    # expected-error @+1 {{positional-only parameter passed as keyword parameter: 'b'}}
    takes_pos_only_param[x, b=x]()

    # expected-error @+1 {{positional-only parameters passed as keyword parameters: 'b', 'a'}}
    takes_pos_only_param[b=x, a=x]()


# expected-note @+1 {{function declared here}}
fn takes_kw_only_param[*, a: int, b: int, c: int = `7`]():
    pass


fn test_missing_kw_only_param[x: int]():
    # TODO: missing kw-only error should take precedence over unknown keyword
    # expected-error @+1 {{unknown keyword parameter: 'd'}}
    takes_kw_only_param[a=x, d=x]()

    # TODO: we should emit an error with a list of expected kwargs here
    # expected-error @+1 {{callee expects 3 parameters, but 0 were specified}}
    takes_kw_only_param[]()

# expected-note @+1 {{function declared here}}
fn takes_kw_only_args(a: int, b: int, *args: int, c: int, d: int = `2`):
    pass

fn test_missing_positional_arg_with_vararg_keyword(x: int):
   # expected-error @+1 {{missing 1 required positional argument: 'b'}}
   takes_kw_only_args(x, c=`2`)

fn test_missing_keyword_arg_with_vararg_keyword(x: int):
   takes_kw_only_args(x, x, c=`2`)
