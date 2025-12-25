# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s

# expected-error @+1 {{unterminated backtick identifier}}
`

# // -----

# expected-error @+1 {{unexpected character}}
!

# // -----

# expected-error @+1 {{leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers}}
0123

# // -----

# expected-error @+1 {{no digits specified for octal literal}}
0o_

# // -----

# expected-error @+1 {{expecting a digit after the exponent}}
1e+*

# // -----

# expected-error @+1 {{expecting a digit after the exponent}}
1e*

# // -----

# expected-error @+1 {{unterminated string}}
"Hello'

# // -----

# expected-error @+1 {{unterminated string}}
'Hello"

# // -----

# expected-error @+1 {{unterminated string}}
"Hello

# // -----

# expected-error @+1 {{unterminated string}}
'Hello

# // -----

# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"A\x4"

# // -----

# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"A\x"

# // -----

# expected-error @+1 {{invalid escape sequence}}
"A\zB"

# // -----

# expected-error @+1 {{unterminated string}}
"AB\"

# // -----

# expected-error @+1 {{unterminated string}}
r"AB\"

# // -----

# Issue #12818
fn inconsistent_indent():
    var x = __mlir_attr.`1 : index`
   	var y = __mlir_attr.`2 : index`  # expected-error {{leading indentation uses inconsistent whitespace (tabs and spaces) than previous line}}
