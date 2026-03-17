# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that @stable(recursive=True) suppresses warnings for methods defined in
# extensions of the imported type, not just methods defined in the struct body.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

@stable(recursive=True)
from test_std_mock import UnstableStruct


def test_extension_method_suppressed():
    # extension_method is defined in a __extension block in the mock package,
    # not in UnstableStruct's body.  The suppression should still apply,
    # the user does not care _where_ the methods are defined.
    var x = UnstableStruct()
    _ = x.extension_method()


# CHECK-NOT: warning: use of unstable API
