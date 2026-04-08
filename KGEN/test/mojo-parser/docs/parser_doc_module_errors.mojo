# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s
# FileModuleOp is located at line 1, col 1 of the buffer, so the diagnostic
# fires there. @-9 counts back from this line to line 1.
# expected-warning @-9 {{public module 'parser_doc_module_errors' is missing a doc string}}

# A public module without a doc string should raise a diagnostic.
# This file intentionally has no module-level doc string.

struct ArgStruct:
    """A stub type for arguments."""
    pass
