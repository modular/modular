# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.dir
# RUN: mojo precompile %S/inputs/link_dep/dotted.dep -o %t.dir/dotted.dep.mojoc
# RUN: mojo precompile -I %t.dir %S/inputs/link_dep/user_pkg -o %t.dir/user_pkg.mojoc
# RUN: mojo run -I %t.dir %s | FileCheck %s

# A precompiled package records its precompiled imports as link dependencies
# by their package *names*. A dependency on a dotted-named package must
# round-trip as a single name, not be re-split as a module path when the
# dependency is resolved from bytecode.

from user_pkg import user_value


def main() raises:
    # CHECK: 42
    print(user_value())
