# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct Foo:
    pass


fn take_instance_param[a: Foo]():
    pass


# expected-note @+1 {{function declared here}}
fn takes_instance_arg(a: Foo):
    pass


# COM: Issue #27654: Parser crash: Assertion failed: Types should match
# COM: https://github.com/modularml/mojo/issues/1607 Improved error message for this common error
fn test_type_instead_of_instance() -> Foo:
    # expected-error @+1 {{cannot pass 'Foo' type value, parameter expected an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    take_instance_param[Foo]
    # expected-error @+1 {{invalid call to 'takes_instance_arg': argument #0 cannot be converted from type value 'Foo' to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    takes_instance_arg(Foo)
    # expected-error @+1 {{cannot implicitly convert 'Foo' type value to an instance of 'Foo' in return value; did you mean to instantiate 'Foo'?}}
    return Foo


# COM: https://github.com/modularml/modular/issues/29438
# COM: ensure we do not crash in the example below, but emit an error.
struct MadeFromPack[*Ts: AnyRegType]:
    fn __init__(inout self, *args: *Ts):
        pass


struct WrapsMadeFromPack[*Ts: AnyRegType]:
    var data: MadeFromPack[Ts]

    fn __init__(inout self, *args: *Ts):
        # expected-error @+1 {{cannot implicitly convert '*Ts' value to 'MadeFromPack[Ts]' in assignment}}
        self.data = args


struct ConvertFromInt:
    fn __init__(inout self, arg: Int):
        pass


fn init_self_conversion():
    # expected-error @below {{cannot implicitly convert 'fn(self = inout ConvertFromInt, /, arg = Int) -> None' value to 'fn() -> None' in alias initializer}}
    alias f: fn () -> None = ConvertFromInt.__init__
