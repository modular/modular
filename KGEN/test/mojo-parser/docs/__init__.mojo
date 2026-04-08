# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s
# FileModuleOp is located at line 1, col 1 of the buffer, so the diagnostic
# fires there. @-9 counts back from this line to line 1.
# expected-warning @-9 {{public module '__init__' is missing a doc string}}

# '__init__' is treated as a public module even though its name begins with '_':
# it is the public package initializer, not a private symbol.
# This file intentionally has no module-level doc string.
