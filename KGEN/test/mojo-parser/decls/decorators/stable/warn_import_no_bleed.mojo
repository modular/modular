# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that @stable(recursive=True) used in a dependency does NOT suppress
# warnings in the importing file. The override is strictly file-scoped.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

# test_std_mock_reexporter imports UnstableStruct with @stable(recursive=True).
# That override lives in the reexporter's file scope only.
from test_std_mock_reexporter import UnstableStruct


def test_type_ref_still_warns():
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var _x: UnstableStruct


def test_constructor_still_warns():
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var x = UnstableStruct()
    # CHECK: warning: use of unstable API 'unstable_method'
    _ = x.unstable_method()
