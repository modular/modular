# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s --verify-diagnostics

# Wrong arguments for decorator errors

# expected-error @below {{@op_implementation expects a string literal argument}}
@op_implementation(3)
struct WrongArg1:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

# expected-error @below {{@op_implementation expects a string literal argument}}
@op_implementation
struct WrongArg2:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

# Already defined error

@op_implementation("custom.already_defined")
struct AlreadyDefined1:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

# expected-error @below {{custom op 'custom.already_defined' is already defined}}
@op_implementation("custom.already_defined")
struct AlreadyDefined2:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

# No `impl` method error

# expected-error @below {{struct annotated with '@op_implementation' should define an `impl` method}}
@op_implementation("custom.no_impl")
struct NoImplOp:
    pass

# Two `impl` methods error

# expected-error @below {{cannot form a reference to overloaded declaration of 'impl'}}
@op_implementation("custom.overloaded_impl")
struct TwoImplsOp:
    @staticmethod
    # expected-note @+1 {{candidate declared here}}
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

    @staticmethod
    # expected-note @+1 {{candidate declared here}}
    fn impl(x: Int64, y: Int64) -> Int64:
        return x + y
