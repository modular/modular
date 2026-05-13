# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package --experimental-fixit %S/test_package_fixit | FileCheck %s --check-prefix=AUTO-FIXIT
# RUN: mojo package --experimental-fixit %S/test_package_fixit | FileCheck %s --check-prefix=NO-FIXIT

# AUTO-FIXIT: Fixits applied.
# NO-FIXIT: No fixits to apply.

# After applying the fixits, the build should succeed.
# RUN: mojo package %S/test_package_fixit -o test_package_fixit.mojoc

# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cat %S/test_package_fixit/__init__.mojo | grep -v "# CHECK" | FileCheck %S/test_package_fixit/__init__.mojo
# RUN: cat %S/test_package_fixit/old_impl.mojo | grep -v "# CHECK" | FileCheck %S/test_package_fixit/old_impl.mojo
