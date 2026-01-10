# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package -Wno-error %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WNO-ERROR
# RUN: not mojo package -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WERROR
# RUN: mojo package -Werror -Wno-error %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WERROR-THEN-WNO
# RUN: not mojo package -Wno-error -Werror %S/test_package_werror 2>&1 | FileCheck %s --check-prefix=WNO-THEN-WERROR

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
