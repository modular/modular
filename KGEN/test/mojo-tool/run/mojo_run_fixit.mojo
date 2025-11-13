# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo run --experimental-fixit %s | FileCheck %s --check-prefix=AUTO-FIXIT
# RUN: mojo run --experimental-fixit %s | FileCheck %s --check-prefix=NO-FIXIT

# AUTO-FIXIT: Fixits applied.
# NO-FIXIT: No fixits to apply.

# After applying the fixits, the build should succeed.
# RUN: mojo run %s

# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cat %s | grep -v "# CHECK" | FileCheck %s


# CHECK-LABEL: fn old_origin_of
fn old_origin_of[T: AnyType](a: T):
    # CHECK-NEXT: _ = origin_of(a)
    # CHECK-NOT: _ = __origin_of(a)
    _ = __origin_of(a)


# CHECK-LABEL: fn old_origin_of_2
fn old_origin_of_2[T: AnyType](b: T):
    # CHECK-NEXT: _ = origin_of(b)
    # CHECK-NOT: _ = __origin_of(b)
    _ = __origin_of(b)


def main():
    old_origin_of(1)
    old_origin_of_2(2)
