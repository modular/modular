# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias AnyRegType = __mlir_type.`!kgen.type`
alias Int = __mlir_type.index

alias `7` = __mlir_attr.`7 : index`


@register_passable
struct VariadicList[type: AnyRegType]:
    alias storage_type = __mlir_type[`!kgen.variadic<`, type, `>`]

    fn __init__(value: Self.storage_type) -> Self:
        return Self {}


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


# expected-note @+1 {{function declared here}}
fn takes_pos_only(a: Int, b: Int, /):
    pass


fn test_pos_only_passed_by_kw(x: Int):
    # expected-error @+1 {{got 1 positional-only argument passed as keyword operand: 'b'}}
    takes_pos_only(x, b=x)

    # expected-error @+1 {{got 2 positional-only arguments passed as keyword operands: 'a', 'b'}}
    takes_pos_only(b=x, a=x)


# expected-note @+1 {{function declared here}}
fn takes_kw_only(*, a: Int, b: Int, c: Int = `7`):
    pass


fn test_missing_kw_only(x: Int):
    # COM: missing kw-only error takes precedence over unknown keyword
    # expected-error @+1 {{missing 1 required keyword-only argument: 'b'}}
    takes_kw_only(a=x, d=x)

    # expected-error @+1 {{missing 2 required keyword-only arguments: 'a', 'b'}}
    takes_kw_only()


# expected-note @+1 {{function declared here}}
fn takes_pos_or_kw(i: Int, j: Int):
    pass


# expected-note @+1 {{function declared here}}
fn var_func(*args: Int):
    pass


# expected-note @+1 {{function declared here}}
fn pack_func[*Ts: AnyRegType](*args: *Ts):
    pass


fn test_unexpected_kw(x: Int):
    # expected-error @+1 {{unknown keyword argument: 'c'}}
    takes_pos_or_kw(x, c=x, j=x)
    # expected-error @+1 {{unknown keyword arguments: 'c', 'd'}}
    takes_pos_or_kw(x, d=x, c=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    var_func(args=x)
    # expected-error @+1 {{unknown keyword argument: 'args'}}
    pack_func(args=x)


fn test_passed_by_pos_and_kw(x: Int):
    # expected-error @+1 {{argument #0 ('i') passed both as positional and keyword operand}}
    takes_pos_or_kw(x, i=x)
