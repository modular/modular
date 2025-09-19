# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test var keyword formatting and spacing."""

import mblack

# Test cases for var keyword spacing
SOURCES = [
    "fn method(var self): pass",
    "fn func(var x: Int): pass",
    "fn method(var self, y: Int): pass",
]

EXPECTED_OUTPUTS = [
"""fn method(var self):
    pass
""",
"""fn func(var x: Int):
    pass
""",
"""fn method(var self, y: Int):
    pass
"""
]

def test_var_keyword_spacing():
    """Test that var keyword is properly formatted with spacing."""

    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    for source, expected in zip(SOURCES, EXPECTED_OUTPUTS):
        assert mblack.format_str(source, mode=mode) == expected
