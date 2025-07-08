# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: mojo doc %s | FileCheck %s


# CHECK: "default": "\"\\n\"",
fn testFn(stringArgument: StaticString = "\n"):
    """Function description text.

    Args:
        stringArgument: Argument description.
    """
    pass
