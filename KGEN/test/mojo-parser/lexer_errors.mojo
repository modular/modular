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
def inconsistent_indent():
    var x = __mlir_attr.`1 : index`
   	var y = __mlir_attr.`2 : index`  # expected-error {{leading indentation uses inconsistent whitespace (tabs and spaces) than previous line}}

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t"Hello
t"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t'Hello
t'Hello'

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"""Hello

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t'''Hello

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Hello{name

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t"Hello
World"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t'Hello
World'

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
T"Hello
T"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
T'Hello
T'Hello'

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Unclosed {expr
t"Unclosed {expr}"

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Missing closing brace {expr"
t"Missing closing brace {expr}"

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Nested {t"inner} incomplete"
t"Nested {t"inner"} incomplete"

# // -----

# Raw t-string: unescaped newline with double quotes (rt prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt"Hello
rt"Hello"

# // -----

# Raw t-string: unescaped newline with single quotes (rt prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt'Hello
rt'Hello'

# // -----

# Raw t-string: unescaped newline with double quotes (tr prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tr"Hello
tr"Hello"

# // -----

# Raw t-string: unescaped newline with single quotes (tr prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tr'Hello
tr'Hello'

# // -----

# Raw t-string: unterminated triple-quoted (rt prefix)
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"""Hello

# // -----

# Raw t-string: unterminated triple-quoted single quotes (rt prefix)
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt'''Hello

# // -----

# Raw t-string: unclosed expression
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Hello{name

# // -----

# Raw t-string: newline in middle of string (double quotes)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt"Hello
World"

# // -----

# Raw t-string: newline in middle of string (single quotes)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt'Hello
World'

# // -----

# Raw t-string: uppercase prefix variants
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
Rt"Hello
Rt"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rT"Hello
rT"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
RT"Hello
RT"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tR"Hello
tR"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
Tr"Hello
Tr"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
TR"Hello
TR"Hello"

# // -----

# Raw t-string: unclosed expression with recovery line
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Unclosed {expr
rt"Unclosed {expr}"

# // -----

# Raw t-string: missing closing brace
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Missing closing brace {expr"
rt"Missing closing brace {expr}"

# // -----

# Raw t-string: nested raw t-string incomplete
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Nested {rt"inner} incomplete"
rt"Nested {rt"inner"} incomplete"
