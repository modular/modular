# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_global_comptime():
    source = (
        "comptime   b =  6\n"
    )
    expected = (
        "comptime b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_nested_comptime():
    source = (
        "struct Foo:\n"
        "    comptime  b =  6\n"
    )
    expected = (
        "struct Foo:\n"
        "    comptime b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if():
    source = (
        "fn foo[a: Bool]():\n"
        "    comptime if a:\n"
        "            var x: Int\n"
    )
    expected = (
        "fn foo[a: Bool]():\n"
        "    comptime if a:\n"
        "        var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if_elif_else():
    source = (
        "fn foo[a: Bool,  b: Bool]():\n"
        "    comptime if a:\n"
        "            var x: Int\n"
        "    elif b:\n"
        "                var y: Int\n"
        "    else:\n"
        "        var z: Int\n"
    )
    expected = (
        "fn foo[a: Bool, b: Bool]():\n"
        "    comptime if a:\n"
        "        var x: Int\n"
        "    elif b:\n"
        "        var y: Int\n"
        "    else:\n"
        "        var z: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_for():
    source = (
        "fn foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "            print(i)\n"
    )
    expected = (
        "fn foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        print(i)\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_for_nested_if():
    source = (
        "fn foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        comptime if True:\n"
        "                    print(i)\n"
    )
    expected = (
        "fn foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        comptime if True:\n"
        "            print(i)\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if_extra_whitespace():
    """Extra whitespace between comptime and if should be normalized."""
    source = (
        "fn foo[x: Int]():\n"
        "    comptime        if x > 0:\n"
        "        var y: Int\n"
    )
    expected = (
        "fn foo[x: Int]():\n"
        "    comptime if x > 0:\n"
        "        var y: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_on_separate_line():
    """comptime on separate line from if is invalid syntax."""
    import mblack
    from mblack import TargetVersion

    source = (
        "fn foo[x: Int]():\n"
        "    comptime\n"
        "    if x > 0:\n"
        "        var y: Int\n"
    )
    mode = mblack.Mode(target_versions={TargetVersion.MOJO})
    try:
        mblack.format_str(source, mode=mode)
        assert False, "Should have raised InvalidInput"
    except mblack.parsing.InvalidInput:
        pass  # Expected


def test_comptime_expr_rhs_of_binop():
    """comptime(expr) on the right-hand side of a binary operator."""
    source = "fn foo():\n" "    var y = 1 + comptime(1 * 2)\n"
    expected = "fn foo():\n" "    var y = 1 + comptime (1 * 2)\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_lhs_of_binop():
    """comptime(expr) on the left-hand side of a binary operator."""
    source = "fn foo():\n" "    var y = comptime(1 * 2) + 1\n"
    expected = "fn foo():\n" "    var y = comptime (1 * 2) + 1\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_as_argument():
    """comptime(expr) as a function argument."""
    source = "fn foo():\n" "    print(comptime(42))\n"
    expected = "fn foo():\n" "    print(comptime (42))\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_multiple():
    """Multiple comptime(expr) in one expression."""
    source = "fn foo():\n" "    var z = comptime(1) + comptime(2)\n"
    expected = "fn foo():\n" "    var z = comptime (1) + comptime (2)\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_in_list_literal():
    """comptime(expr) inside a list literal."""
    source = "fn foo():\n" "    var x = [comptime(1), comptime(2)]\n"
    expected = "fn foo():\n" "    var x = [comptime (1), comptime (2)]\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_in_if_condition():
    """comptime(expr) used as a condition in an if statement."""
    source = (
        "fn foo():\n"
        "    if comptime(thread_layout.rank() == 1):\n"
        "        pass\n"
    )
    expected = (
        "fn foo():\n"
        "    if comptime (thread_layout.rank() == 1):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_illegal_keyword():
    """comptime with illegal keyword should be rejected."""
    import mblack
    from mblack import TargetVersion

    source = (
        "fn foo():\n"
        "    comptime try:\n"
        "        var x: Int\n"
    )
    mode = mblack.Mode(target_versions={TargetVersion.MOJO})
    try:
        mblack.format_str(source, mode=mode)
        assert False, "Should have raised InvalidInput"
    except mblack.parsing.InvalidInput:
        pass  # Expected
