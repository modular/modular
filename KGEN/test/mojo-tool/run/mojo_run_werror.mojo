# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo run -Wno-error %s 2>&1 | FileCheck %s --check-prefix=WNO-ERROR
# RUN: not mojo run -Werror %s 2>&1 | FileCheck %s --check-prefix=WERROR
# RUN: mojo run -Werror -Wno-error %s 2>&1 | FileCheck %s --check-prefix=WERROR-THEN-WNO
# RUN: not mojo run -Wno-error -Werror %s 2>&1 | FileCheck %s --check-prefix=WNO-THEN-WERROR

# WNO-ERROR: warning: assignment to 'foo' was never used
# WNO-ERROR-NOT: error: assignment to 'foo' was never used

# WERROR: error: assignment to 'foo' was never used
# WERROR-NOT: warning: assignment to 'foo' was never used

# -Werror followed by -Wno-error: warnings remain warnings (last wins)
# WERROR-THEN-WNO: warning: assignment to 'foo' was never used
# WERROR-THEN-WNO-NOT: error: assignment to 'foo' was never used

# -Wno-error followed by -Werror: warnings become errors (last wins)
# WNO-THEN-WERROR: error: assignment to 'foo' was never used
# WNO-THEN-WERROR-NOT: warning: assignment to 'foo' was never used


def main():
    var foo = 1
