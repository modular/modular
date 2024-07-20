# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct Foo:
    fn __init__(inout self): pass


fn take_instance_param[a: Foo]():
    pass


# expected-note @+1 {{function declared here}}
fn takes_instance_arg(a: Foo):
    pass


# COM: Issue #27654: Parser crash: Assertion failed: Types should match
# COM: https://github.com/modularml/mojo/issues/1607 Improved error message for this common error
fn test_type_instead_of_instance() -> Foo:
    # expected-error @+1 {{cannot implicitly convert 'Foo' type as a value to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    take_instance_param[Foo]
    # expected-error @+1 {{invalid call to 'takes_instance_arg': argument #0 cannot be converted from type value 'Foo' to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    takes_instance_arg(Foo)
    # expected-error @+1 {{cannot implicitly convert 'Foo' type as a value to an instance of 'Foo'; did you mean to instantiate 'Foo'?}}
    return Foo


# COM: https://github.com/modularml/modular/issues/29438
# COM: ensure we do not crash in the example below, but emit an error.
struct MadeFromPack[*Ts: AnyType]:
    fn __init__(inout self, *args: *Ts):
        pass


struct WrapsMadeFromPack[*Ts: AnyType]:
    var data: MadeFromPack[Ts]

    fn __init__(inout self, *args: *Ts):
        # expected-error @+1 {{cannot implicitly convert 'VariadicPack[0, args, AnyType, Ts]' value to 'MadeFromPack[Ts]'}}
        self.data = args


struct Constructible:
    fn __init__(inout self, arg: Int):
        pass


fn init_self_conversion():
    # expected-error @below {{cannot implicitly convert 'fn(inout self: Constructible, /, arg: Int) -> None' value to 'fn() -> None'}}
    alias f: fn () -> None = Constructible.__init__


@value
struct ConvertibleFromInt:
    fn __init__(inout self, arg: Int):
        pass


@value
# expected-note @below {{candidate generated with type 'fn(inout AmbiguousCtor, /, owned a: ConvertibleFromInt, b: Int) -> None'}}
struct AmbiguousCtor:
    var a: ConvertibleFromInt
    var b: Int

    # expected-note @below {{candidate declared here}}
    fn __init__(inout self, b: Int, a: ConvertibleFromInt):
        pass


struct AlsoConvertibleFromInt:
    fn __init__(inout self, arg: Int):
        pass


struct AmbiguousConversion:
    # expected-note @below {{candidate declared here}}
    fn __init__(inout self, x: ConvertibleFromInt):
        pass

    # expected-note @below {{candidate declared here}}
    fn __init__(inout self, x: AlsoConvertibleFromInt):
        pass


fn ambiguous_ctor_call(x: Int):
    # expected-error @below {{ambiguous call}}
    AmbiguousCtor(x, x)

    # expected-error @below {{ambiguous call to '__init__', each candidate requires 1 implicit conversion}}
    AmbiguousConversion(x)


# MOCO-990: Conditional conformance trick fails on SIMD constructor from Bool
struct MySIMD[value: Int]:
    fn __init__(inout self: MySIMD[0], value: MyBool):
        pass
struct MyBool:
    fn __init__(inout self, value: MySIMD[0]):
        pass
fn test_bad_conversion(a: MySIMD[0]):
    # expected-error @+1 {{cannot implicitly convert 'MySIMD[0]' value to 'MySIMD[1]'}}
    var b : MySIMD[1] = a
