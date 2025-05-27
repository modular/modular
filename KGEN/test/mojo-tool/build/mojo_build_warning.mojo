# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build %s 2>&1 | FileCheck %s --check-prefix=WITH-WARNINGS
# RUN: %mojo-build %s --disable-warnings 2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT-WARNINGS


# This warning is issued via the Diags class.
#
# WITH-WARNINGS: warning: Use bar instead
# WITHOUT-WARNINGS-NOT: warning: Use bar instead
@deprecated("Use bar instead")
fn deprecatedFn():
    pass


# This warning is issued via the MLIR diagnostic engine
#
# WITH-WARNINGS: warning: assignment to 'x' was never used
# WITHOUT-WARNINGS-NOT: warning: assignment to 'x' was never used
fn unusedLocal():
    var x = 42


fn main():
    deprecatedFn()
    unusedLocal()
    return
