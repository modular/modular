# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_ref_origin_in_parameter():
    """No space between ref and [origin] in parameters."""
    source = "def foo(ref[MutAnyOrigin] r: Int): pass"
    expected = (
        "def foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_parameter_with_space():
    """Space between ref and [origin] should be removed."""
    source = "def foo(ref [MutAnyOrigin] r: Int): pass"
    expected = (
        "def foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type():
    """No space between ref and [origin] in return types."""
    source = "def bar(x: String) raises Int -> ref[x] String: return x"
    expected = (
        "def bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type_with_space():
    """Space between ref and [origin] in return types should be removed."""
    source = "def bar(x: String) raises Int -> ref [x] String: return x"
    expected = (
        "def bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_multiple_origins():
    """Multiple origins should be formatted correctly."""
    source = "def foo[a: Origin, b: Origin](ref[a, b] r: Int): pass"
    expected = (
        "def foo[a: Origin, b: Origin](ref[a, b] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_with_space_before_param():
    """Ensure space is preserved between ] and parameter name."""
    source = "def foo[origin: Origin](ref[origin]r: Int): pass"
    expected = (
        "def foo[origin: Origin](ref[origin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param():
    """Space after ref[origin] with untyped parameter (e.g., self)."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED]self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_space():
    """Preserve correct spacing for untyped parameter."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_extra_spaces():
    """Fix extra space between ref and [origin] for untyped parameter."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref [AddressSpace.SHARED] self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_out_address_space_in_initializer():
    """No space between out and [address_space] in initializers."""
    source = "def __init__(out[addrspace] self): pass"
    expected = "def __init__(out[addrspace] self):\n    pass\n"
    assert_mojo_format(source, expected)


def test_out_address_space_in_initializer_with_space():
    """Space between out and [address_space] should be removed."""
    source = "def __init__(out [addrspace] self): pass"
    expected = "def __init__(out[addrspace] self):\n    pass\n"
    assert_mojo_format(source, expected)
