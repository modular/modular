# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

TEST_CASE = """struct Foo:
    @deprecated(   "abc")
    alias b =  6
"""

EXPECTED_OUTPUT = """struct Foo:
    @deprecated("abc")
    alias b = 6
"""


def test_decorated_alias():
    import mblack

    result = mblack.format_str(TEST_CASE, mode=mblack.FileMode())
    assert result == EXPECTED_OUTPUT


if __name__ == "__main__":
    test_decorated_alias()
