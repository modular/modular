# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test brace initializers with keyword arguments."""

def test_brace_initializer_simple():
    """Test that brace initializer syntax can be parsed without errors."""
    import mblack

    # Test cases focusing on the brace initializer expressions
    test_cases = [
        "obj{field=value}",
        "obj{field1=value1, field2=value2}",
        "result = foo{3, y=4}",  # Python-valid version
        "x = SomeType{x=1, y=2}"
    ]

    for source in test_cases:
        try:
            # Just verify it doesn't crash during formatting
            mblack.format_str(source, mode=mblack.FileMode())
            print(f"✓ Passed: {source}")
        except Exception as e:
            print(f"✗ Failed: {source} - {e}")
            raise AssertionError(f"Failed to format: {source}")


if __name__ == "__main__":
    test_brace_initializer_simple()
    print("All tests passed!")
