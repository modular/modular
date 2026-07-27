# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo-lsp-simple-client --fail-on-diagnostics %s > %t 2>&1; true
# RUN: FileCheck %s < %t

# A docstring code block is parsed in a synthetic wrapper buffer (a distinct
# buffer from this file), but an import of this module written inside it is
# still a self-import of the module the wrapper wraps, and must be rejected as
# one rather than resolving the module into its own docstring example.
#
# CHECK: module 'docstring_self_import' cannot import itself


def example():
    """Example whose code block self-imports the enclosing module.

    ```mojo
    import docstring_self_import
    ```
    """
    pass
