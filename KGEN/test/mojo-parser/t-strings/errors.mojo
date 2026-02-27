# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s

# =============================================================================
# Empty and malformed expressions
# =============================================================================

fn test_empty_braces():
    # Empty expression (just braces)
    # expected-error @below {{t-string expression cannot be empty}}
    _ = t"Hello {}"

# =============================================================================
# Unterminated errors
# =============================================================================

fn test_newline_in_single_quote():
    # expected-error @below {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
    _ = t"Hello
    # expected-error @below {{unterminated string}}
    World"

# =============================================================================
# Lone closing braces
# =============================================================================
# Note: With single-token lexing, a lone `}` at brace depth 0 is included as
# literal text (not an error). Use `}}` to write a literal brace character.

fn test_lone_brace():
    # Lone closing brace is treated as literal text (no error).
    _ = t"Hello }"


fn test_triple_quote_lone_brace():
    # Lone closing brace in triple-quoted t-string is literal text.
    _ = t"""Hello }"""


fn test_multiple_lone_braces():
    # Multiple lone closing braces are literal text.
    _ = t"Hello } World }"


fn test_escaped_brace_after_expression():
    var value = 42
    # `}}` after expression close is an escaped literal brace (not an error).
    _ = t"Value: {value}}"


fn test_lone_brace_with_nested_quotes():
    # Lone `}` with nested quotes is literal text.
    _ = t"Hello } 'nested string' end"


# =============================================================================
# Format specs (not yet supported)
# =============================================================================

fn test_format_spec_with_precision():
    var x = 42
    # expected-error @below {{format specs are not yet supported in t-strings}}
    _ = t"{x:.2}"


fn test_format_spec_empty():
    var x = 42
    # expected-error @below {{format specs are not yet supported in t-strings}}
    _ = t"{x:}"


fn test_format_spec_in_multiple_interpolations():
    var x = 42
    var y = 3.14159
    # expected-error @below {{format specs are not yet supported in t-strings}}
    _ = t"x={x:}, y={y:.3f}"


fn test_format_spec_with_complex_expression():
    var x = 10
    # expected-error @below {{format specs are not yet supported in t-strings}}
    _ = t"{x + 10:.2}"


# =============================================================================
# expression errors
# =============================================================================

fn test_undefined_variable():
    # Test: Undefined variable in interpolation
    # expected-error @below {{use of unknown declaration 'undefined_var'}}
    var s1 = t"Hello, {undefined_var}!"


fn test_invalid_expression():
    # Test: Invalid expression in interpolation (bare operator)
    # expected-error @below {{unexpected token in expression}}
    var s2 = t"Result: {+}"


fn test_incomplete_expression():
    # Test: Incomplete expression in interpolation
    # expected-error @below {{unexpected token in expression}}
    var s3 = t"Value: {1 +}"


fn test_multiple_undefined_vars():
    # Test: Multiple undefined variables in one t-string (only first error reported)
    # expected-error @below {{use of unknown declaration 'x'}}
    var s4 = t"Values: {x} and {y}"

# =============================================================================
# comptime errors
# =============================================================================

fn test_unmaterializable_value_in_tstring():
    comptime l = [1, 2, 3]
    # expected-error @below {{cannot materialize comptime value of type 'List[Int]' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note @below {{use 'materialize' to explicitly materialize the value}}
    _ = t"List: {l}"

fn test_runtime_value_in_comptime_tstring():
    var x = 42
    # expected-error @below {{cannot use dynamic value in comptime initializer}}
    comptime tstring = t"Value: {x}"

# =============================================================================
# Catastrophic errors (these must be last - they break parser recovery)
# =============================================================================

fn test_unclosed_expression():
    # Unclosed expression (missing closing })
    # With single-token lexing, the lexer sees the quote inside the expression
    # as starting a nested string, ultimately leading to an unterminated t-string.
    # expected-error @below {{unterminated t-string (missing closing quote)}}
    _ = t"Hello {name"
