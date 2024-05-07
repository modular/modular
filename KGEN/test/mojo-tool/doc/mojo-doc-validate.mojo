# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Validation itself is tested elsewhere; here we test only that the driver
# passes the `-validate` option through to the parser (this file contains
# validation warnings).
# RUN: mojo doc -diagnose-missing-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARN
# CHECK-WARN: mojo-doc-validate.mojo:{{.*}}warning: unknown argument
# RUN: not mojo doc -diagnose-missing-doc-strings -validate-doc-strings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-VALIDATE
# CHECK-VALIDATE: mojo-doc-validate.mojo:{{.*}}error: unknown argument
# CHECK-VALIDATE-NOT: "name": "mojo-doc-validate",


fn f(x: Int):
    """This is an invalid doc string.

    Args:
      y: This argument doesn't appear in the argument list.
      z: Neither does this one.
    """
    pass
