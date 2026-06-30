# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests that the Mojo line-continuation normalizer counts only brackets that
# appear in *code*.  An unbalanced bracket inside a comment, a string literal,
# or a docstring must not desync the normalizer's bracket-depth tracking and
# silently disable line rejoining for the rest of the scope.

from tests.util import assert_mojo_format


def test_unmatched_paren_in_comment_does_not_disable_continuation():
    """A "(" in a comment must not stop a later continuation from rejoining."""
    source = (
        "def main():\n"
        "    # returns a (partial result\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    # returns a (partial result\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_string_does_not_disable_continuation():
    """A "(" inside a string literal must not stop a later continuation."""
    source = (
        "def main():\n"
        '    var s = "an unmatched ( paren in a string"\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    var s = "an unmatched ( paren in a string"\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_oneline_docstring_does_not_disable_continuation():
    """A "(" in a one-line docstring must not desync bracket tracking."""
    source = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren."""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren."""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_multiline_docstring_does_not_disable_continuation():
    """The original repro: a "(" on a multi-line docstring's summary line whose
    matching ")" lives inside the (uncounted) string body."""
    source = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren.\n'
        "\n"
        "    More body text (with a balanced pair) here.\n"
        '    """\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren.\n'
        "\n"
        "    More body text (with a balanced pair) here.\n"
        '    """\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)
