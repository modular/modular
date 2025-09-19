# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import mblack

SOURCE = """struct Foo:
    @deprecated(   "abc")
    alias b =  6
"""

EXPECTED_OUTPUT = """struct Foo:
    @deprecated("abc")
    alias b = 6
"""

def test_decorated_alias():
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    assert mblack.format_str(SOURCE, mode=mode) == EXPECTED_OUTPUT
