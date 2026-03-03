# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --experimental-fixit %s | FileCheck %s --check-prefix=AUTO-FIXIT
# RUN: %mojo-build --experimental-fixit %s | FileCheck %s --check-prefix=NO-FIXIT

# AUTO-FIXIT: Fixits applied.
# NO-FIXIT: No fixits to apply.

# After applying the fixits, the build should succeed.
# RUN: %mojo-build %s

# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cat %s | grep -v "# CHECK" | FileCheck %s

# CHECK: from std.collections import List
from collections import List

def main():
    var arguments = List[String]()
    arguments.append("1")
    arguments.append("2")

    print(arguments)
