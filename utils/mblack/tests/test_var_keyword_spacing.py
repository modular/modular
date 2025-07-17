# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test var keyword formatting and spacing."""

def test_var_keyword_spacing():
    """Test that var keyword is properly formatted with spacing."""
    import mblack

    # Test cases for var keyword spacing
    test_cases = [
        "fn method(var self): pass",
        "fn func(var x: Int): pass",
        "fn method(var self, y: Int): pass",
    ]

    for source in test_cases:
        try:
            result = mblack.format_str(source, mode=mblack.FileMode())
            # Check that 'var' is not merged with the following identifier
            assert "varself" not in result, f"'var' merged with identifier in: {source}"
            assert "varx" not in result, f"'var' merged with identifier in: {source}"
            assert "varother" not in result, f"'var' merged with identifier in: {source}"
            # Ensure proper spacing
            assert "var self" in result or "var x" in result, f"Missing proper 'var' spacing in: {source}"
            print(f"✓ Passed: {source}")
        except Exception as e:
            print(f"✗ Failed: {source} - {e}")
            raise AssertionError(f"Failed to format: {source}")


if __name__ == "__main__":
    test_var_keyword_spacing()
    print("All var keyword spacing tests passed!")
