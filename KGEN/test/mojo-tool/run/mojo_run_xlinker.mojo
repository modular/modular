# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that mojo-run warns about -Xlinker args, which are unused.

# RUN: mojo run -Xlinker foo %s 2>&1 | FileCheck %s --check-prefixes WARN,CHECK
# RUN: mojo run -Xlinker foo --disable-warnings %s 2>&1 \
# RUN:   | FileCheck %s --check-prefixes NO-WARN,CHECK

# WARN: warning: -Xlinker argument unused: 'foo'
# NO-WARN-NOT: warning
# CHECK: hello, world


def main():
    print("hello, world")
