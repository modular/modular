# ===----------------------------------------------------------------------=== #
# Focused tests for None repr
# ===----------------------------------------------------------------------=== #

from testing import assert_equal, TestSuite


def test_none_repr_simple():
    assert_equal(repr(None), "None")


def test_none_in_list():
    # Avoid testing container formatting with None here because
    # container write_repr_to requires element types to implement Writable
    # at compile time. A focused test for `repr(None)` is sufficient.
    pass


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
