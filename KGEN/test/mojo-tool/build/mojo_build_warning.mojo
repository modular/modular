# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build %s 2>&1 | FileCheck %s --check-prefix=WITH-WARNINGS
# RUN: %mojo-build %s --disable-warnings 2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT-WARNINGS


# WITH-WARNINGS: warning: Use bar instead
# WITHOUT-WARNINGS-NOT: warning: Use bar instead
@deprecated("Use bar instead")
fn foo():
    pass


fn main():
    foo()
    return
