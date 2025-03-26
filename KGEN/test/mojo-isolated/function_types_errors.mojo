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
    # expected-error @below {{invalid call to 'infer_variadic': callee parameter #1 has 'fn(x: Int, y: Int, *args: *) -> None' type, but value has type 'fn(i: Int) -> None'}}
    infer_variadic[device_func]()


# // -----


# Tests that we correctly match each incoming argument type against the
# callee's variadic's element trait.

# expected-error @below {{struct 'ZInt' does not implement all requirements for 'Sprongling'}}
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


fn test_infer_variadic():
    # expected-error @below {{invalid call to 'infer_variadic': callee parameter #1 has 'fn(*args: *) -> None' type, but value has type 'fn(i: ZInt, j: ZInt) -> None'}}
    infer_variadic[device_func]()
