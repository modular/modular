# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file %s -verify-diagnostics


struct MemType:
    pass


fn mut_ship_function(mut x: MemType):
    ...


# We can convert from fn(read MemType)->None to fn(mut MemType)->None but not
# vice versa (see TTSMFS).
# expected-error @below {{cannot implicitly convert 'fn(mut x: MemType) -> None' value to 'fn(MemType) -> None' in alias initializer}}
alias read_ship_fn_alias: fn (read MemType) -> None = mut_ship_function



# // -----


# Tests that we detect too few arguments when doing function conversions.

# expected-note @below {{function declared here}}
fn infer_variadic[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
    func: fn(x: Int, y: Int, *args: *ArgTypes)->None
]():
    pass

fn device_func(i: Int):
    pass

fn test_infer_variadic():
    # expected-error @below {{invalid call to 'infer_variadic': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    infer_variadic[device_func]()


# // -----


# Tests that we correctly match each incoming argument type against the
# callee's variadic's element trait.

# expected-note @below {{struct 'ZInt' does not implement all requirements for 'Sprongling'}}
struct ZInt:
    pass

# expected-note @below {{trait 'Sprongling' declared here}}
trait Sprongling:
    # expected-note @below {{required function 'sprongle' is not implemented}}
    fn sprongle(self):
        ...

# expected-note @below {{function declared here}}
fn infer_variadic[
    ArgTypes: __mlir_type[`!kgen.variadic<`, Sprongling, `>`], //,
    func: fn(*args: *ArgTypes)->None
]():
    pass

fn device_func(i: ZInt, j: ZInt):
    pass


# expected-error @below {{cannot bind type 'ZInt' to trait 'Sprongling'}}
fn test_infer_variadic():
    # expected-error @below {{invalid call to 'infer_variadic': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    infer_variadic[device_func]()


# // -----



@register_passable("trivial")
trait ZAnyRPTrait:
    pass

fn device_func(a: Int, b: Bool) -> Int:
    return 73

@value
struct DeviceFunction[*ArgTypes: ZAnyRPTrait]:
    # expected-note @below {{function declared here}}
    fn call(self, *args: *ArgTypes) -> Int:
        return 91

fn compile[
    ArgTypes: __mlir_type[`!kgen.variadic<`, ZAnyRPTrait, `>`], //,
    func: fn(*args: *ArgTypes)->Int
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()

fn main():
    var thing = compile[device_func]()
    # expected-error @below {{invalid call to 'call': method argument #1 cannot be converted from 'StringLiteral["hello"]' to 'Bool'}}
    var result2 = thing.call(42, "hello")
