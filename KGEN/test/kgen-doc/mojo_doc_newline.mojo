# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# RUN: kgen-doc %s | FileCheck %s


# CHECK: "default": "StringSpan(\"\\n\")",
def testFn(stringArgument: StaticString = "\n"):
    """Function description text.

    Args:
        stringArgument: Argument description.
    """
    pass
