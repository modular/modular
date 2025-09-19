# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import mblack

# Test cases focusing on the brace initializer expressions

SOURCES = [
    "obj{field=value}",
    "obj{field1=value1, field2=value2}",
    "result = foo{3, y=4}",  # Python-valid version
    "x = SomeType{x=1, y=2}"
]

EXPECTED_OUTPUTS = [
    "obj {field = value}\n",
    "obj {field1 = value1, field2 = value2}\n",
    "result = foo {3, y = 4}\n",
    "x = SomeType {x = 1, y = 2}\n"
]

def test_brace_initializer_simple():
    """Test that brace initializer syntax can be parsed without errors."""

    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    for source, expected in zip(SOURCES, EXPECTED_OUTPUTS):
        assert mblack.format_str(source, mode=mode) == expected
