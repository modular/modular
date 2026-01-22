# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests for @deprecated decorator warning emission.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===----------------------------------------------------------------------=== #
# Test declarations
# ===----------------------------------------------------------------------=== #


@deprecated("deprecated struct")
# expected-note @below {{'DeprecatedStruct' declared here}}
struct DeprecatedStruct:
    pass


struct NormalStruct:
    pass


@deprecated("deprecated function")
# expected-note @below {{'deprecated_fn' declared here}}
fn deprecated_fn():
    pass


fn normal_fn():
    pass


@deprecated("deprecated trait")
# expected-note @below {{'DeprecatedTrait' declared here}}
trait DeprecatedTrait:
    pass


trait NormalTrait:
    pass


@deprecated("deprecated alias")
# Note: The note format for aliases includes a suffix like `0x, so we use
# expected-note-re to match it. See MOCO-3108.
# expected-note-re @below {{'deprecated_alias{{.*}}' declared here}}
comptime deprecated_alias = 42


comptime normal_alias = 42


struct StructWithDeprecatedMembers:
    # Note: @deprecated on fields is not supported - decorators cannot be applied
    # to `var` statements. This is a parser-level limitation.

    fn __init__(out self):
        pass

    @deprecated("deprecated method")
    # expected-note @below {{'deprecated_method' declared here}}
    fn deprecated_method(self):
        pass

    fn normal_method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated struct as type annotation
# ===----------------------------------------------------------------------=== #


# expected-warning @below {{deprecated struct}}
fn use_deprecated_struct_in_signature(value: DeprecatedStruct):
    pass


fn use_normal_struct_in_signature(value: NormalStruct):
    # No warning expected.
    pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function call
# ===----------------------------------------------------------------------=== #


fn test_deprecated_function_call():
    # expected-warning @below {{deprecated function}}
    deprecated_fn()

    # No warning expected.
    normal_fn()


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function reference (not call)
# ===----------------------------------------------------------------------=== #


fn takes_fn_ref(f: fn () -> None):
    f()


fn test_deprecated_function_reference():
    # Taking a reference to a deprecated function also emits the deprecation warning.
    # expected-warning @below {{deprecated function}}
    var f = deprecated_fn
    takes_fn_ref(f)

    # No warning expected.
    var g = normal_fn
    takes_fn_ref(g)


# ===----------------------------------------------------------------------=== #
# Test: Deprecated trait conformance
# ===----------------------------------------------------------------------=== #


# expected-warning @below {{deprecated trait}}
struct StructConformingToDeprecatedTrait(DeprecatedTrait):
    pass


struct StructConformingToNormalTrait(NormalTrait):
    # No warning expected.
    pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated alias usage
# ===----------------------------------------------------------------------=== #


fn test_deprecated_alias():
    # expected-warning @below {{deprecated alias}}
    _ = deprecated_alias

    # No warning expected.
    _ = normal_alias


# ===----------------------------------------------------------------------=== #
# Test: Deprecated method call
# ===----------------------------------------------------------------------=== #


fn test_deprecated_method_call():
    var obj = StructWithDeprecatedMembers()

    # expected-warning @below {{deprecated method}}
    obj.deprecated_method()

    # No warning expected.
    obj.normal_method()


# ===----------------------------------------------------------------------=== #
# Test: Deprecated method reference (not call)
# ===----------------------------------------------------------------------=== #

fn test_deprecated_method_reference():
    # Taking a reference to a deprecated method emits the deprecation warning.
    # expected-warning @below {{deprecated method}}
    _ = StructWithDeprecatedMembers.deprecated_method

    # No warning expected.
    _ = StructWithDeprecatedMembers.normal_method


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function overload
# ===----------------------------------------------------------------------=== #


@deprecated("deprecated overload")
# expected-note @below {{'overloaded_fn' declared here}}
fn overloaded_fn():
    pass


# expected-warning @below {{deprecated struct}}
fn overloaded_fn(value: DeprecatedStruct):
    pass


fn test_deprecated_function_overload():
    # expected-warning @below {{deprecated overload}}
    overloaded_fn()


# ===----------------------------------------------------------------------=== #
# Test: @deprecated decorator errors
# ===----------------------------------------------------------------------=== #


# expected-error @below {{@deprecated requires a warning message}}
@deprecated
fn no_message():
    pass


comptime NOT_A_STRING = 123


# expected-error @below {{'reason' argument must be a string literal}}
@deprecated(NOT_A_STRING)
fn deprecated_with_non_string_reason():
    pass


# expected-error @below {{'reason' argument must be a string literal}}
@deprecated(reason=NOT_A_STRING)
fn deprecated_with_non_string_keyword_reason():
    pass


# ===----------------------------------------------------------------------=== #
# Test: @deprecated and @stable mutual exclusivity
# ===----------------------------------------------------------------------=== #


# Both decorators cannot be used together.
@deprecated("use something else")
# expected-error @below {{@deprecated and @stable cannot be used together}}
@stable
fn deprecated_and_stable():
    pass


# Order does not matter - still an error.
@stable
# expected-error @below {{@deprecated and @stable cannot be used together}}
@deprecated("use something else")
fn stable_and_deprecated():
    pass


# Another decorator in between does not matter - still an error.
@deprecated("use something else")
@no_inline
# expected-error @below {{@deprecated and @stable cannot be used together}}
@stable
fn deprecated_other_stable():
    pass
