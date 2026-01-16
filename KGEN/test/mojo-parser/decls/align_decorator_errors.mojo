# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Test error cases for @align decorator.

# expected-error @+1 {{@align value must be a positive power of 2}}
@align(0)
struct ZeroAlign:
    var x: Int


# -1 is parsed as unary negation, not an integer literal
# expected-error @+1 {{@align value must be a positive power of 2}}
@align(-1)
struct NegativeAlign:
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(3)
struct NotPowerOfTwo1:
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(5)
struct NotPowerOfTwo2:
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(6)
struct NotPowerOfTwo3:
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(7)
struct NotPowerOfTwo4:
    var x: Int


# expected-error @+1 {{@align value must be a positive power of 2}}
@align(100)
struct NotPowerOfTwo5:
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align
struct MissingArgument:
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align()
struct EmptyArguments:
    var x: Int


# expected-error @+1 {{@align requires exactly one argument}}
@align(64, 128)
struct TooManyArguments:
    var x: Int


# expected-error @+1 {{@align requires a compile-time integer literal}}
@align("64")
struct StringArgument:
    var x: Int


# expected-error @+1 {{@align value exceeds maximum alignment (2^29)}}
@align(1073741824)  # 2^30, exceeds maximum
struct ExcessiveAlignment:
    var x: Int


# expected-error @+1 {{@align value exceeds maximum alignment (2^29)}}
@align(4294967296)  # 2^32, exceeds maximum
struct ExcessiveAlignment2:
    var x: Int


# @align(1) is allowed without warning - useful for parametric alignment fallback
@align(1)
struct AlignOne:
    var x: Int
