# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_ref_origin_in_parameter():
    """No space between ref and [origin] in parameters."""
    source = "fn foo(ref[MutAnyOrigin] r: Int): pass"
    expected = (
        "fn foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_parameter_with_space():
    """Space between ref and [origin] should be removed."""
    source = "fn foo(ref [MutAnyOrigin] r: Int): pass"
    expected = (
        "fn foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type():
    """No space between ref and [origin] in return types."""
    source = "fn bar(x: String) raises Int -> ref[x] String: return x"
    expected = (
        "fn bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type_with_space():
    """Space between ref and [origin] in return types should be removed."""
    source = "fn bar(x: String) raises Int -> ref [x] String: return x"
    expected = (
        "fn bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_multiple_origins():
    """Multiple origins should be formatted correctly."""
    source = "fn foo(ref[a, b] r: Int): pass"
    expected = (
        "fn foo(ref[a, b] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_with_space_before_param():
    """Ensure space is preserved between ] and parameter name."""
    source = "fn foo(ref[origin]r: Int): pass"
    expected = (
        "fn foo(ref[origin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param():
    """Space after ref[origin] with untyped parameter (e.g., self)."""
    source = "fn barriers(ref[AddressSpace.SHARED]self): pass"
    expected = (
        "fn barriers(ref[AddressSpace.SHARED] self):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_space():
    """Preserve correct spacing for untyped parameter."""
    source = "fn barriers(ref[AddressSpace.SHARED] self): pass"
    expected = (
        "fn barriers(ref[AddressSpace.SHARED] self):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_extra_spaces():
    """Fix extra space between ref and [origin] for untyped parameter."""
    source = "fn barriers(ref [AddressSpace.SHARED] self): pass"
    expected = (
        "fn barriers(ref[AddressSpace.SHARED] self):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
