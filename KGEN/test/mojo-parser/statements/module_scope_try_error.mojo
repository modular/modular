# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that 'try' statement at module scope emits an error.
# This is in a separate file because after 'try' is rejected, the 'except'
# keyword becomes an invalid token at module scope, causing a cascading error.

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# expected-error @below {{'try' must be contained in a function}}
try:
    pass
# expected-error @below {{unexpected token in expression}}
except:
    pass
