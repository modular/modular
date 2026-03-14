# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Mock reexporter package for testing that @stable(recursive=True) does not
# bleed across module boundaries.
#
# This module imports UnstableStruct with @stable(recursive=True), suppressing
# stability warnings within THIS file. That override must not propagate to
# files that import from this module.

@stable(recursive=True)
from test_std_mock import UnstableStruct
