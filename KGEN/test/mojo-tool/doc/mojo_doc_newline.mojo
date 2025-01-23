# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: mojo doc %s | FileCheck %s

from utils import StaticString


# CHECK: "default": "StringSlice(\"\\n\")",
fn testFn(stringArgument: StaticString = "\n"):
    """Function description text.

    Args:
        stringArgument: Argument description.
    """
    pass
