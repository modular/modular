# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
This module tests docstring formatting issues."""

# RUN: mojo doc %s | FileCheck %s

# Check that no diagnostics are output:
# RUN: mojo doc %s 2>&1 | FileCheck %s --allow-empty --check-prefix CHECK-DIAG
# CHECK-DIAG-NOT: warning


# CHECK:        "description": "With multiple lines!"
# CHECK:        "name": "MULTILINE_ALIAS"
# CHECK:        "summary": "Docstring for the alias."


alias MULTILINE_ALIAS = 5
"""Docstring for the alias.

With multiple lines!"""

# CHECK-LABEL:  "name": "fn_with_good_format"
# CHECK:   "description": "This is some kind of description.\n\nNotes:\n\n- This is a note.\n\nExample:\n\n```mojo\nprint(\"Example\")\n```\n",


fn fn_with_good_format() -> Int:
    """Docstring with properly formatted notes and examples.

    This is some kind of description.

    Notes:

    - This is a note.

    Example:

    ```mojo
    print("Example")
    ```

    Returns:
        An Int.
    """
    return 33


# CHECK-LABEL:  "name": "fn_with_bad_format",
# CHECK:      "description": "This is some kind of description.\n\nNotes:\n\n- This is a note.\n\nExample:\n\n```mojo\nprint(\"Example\")\n```\n"


fn fn_with_bad_format() -> Int:
    """Docstring with improperly formatted notes and examples.

    This is some kind of description.

    Notes:

        - This is a note.

    Example:

        ```mojo
        print("Example")
        ```

    Returns:
        An Int.
    """
    return 33


# CHECK-LABEL:  "name": "multiline_param_arg_docstrings",
# CHECK:    "overloads": [
# CHECK:      "args": [
# CHECK:        "description": "This argument takes either:\n    - 0: A fish.\n    - 1: A wombat."
# CHECK:        "name": "arg1",
# CHECK:      "parameters": [
# CHECK:        "description": "This parameter takes either:\n    - 0: A cat.\n    - 1: A dog."
# CHECK:        "name": "param1"


fn multiline_param_arg_docstrings[param1: Int](arg1: Int):
    """Docstring for the function.

    This is some kind of description.

    Parameters:
        param1: This parameter takes either:
            - 0: A cat.
            - 1: A dog.

    Args:
        arg1: This argument takes either:
            - 0: A fish.
            - 1: A wombat.
    """
