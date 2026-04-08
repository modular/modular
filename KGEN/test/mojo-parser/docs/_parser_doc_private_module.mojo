# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s

# A private module (name starts with '_') should not trigger the
# missing-doc-string diagnostic, even with no module-level doc string.
