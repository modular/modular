# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests for closure capture lists (`{ captures }`) on `def`.

import pytest
from tests.util import assert_mojo_format

# Bare capture modes (no name). `ref` always requires a name, so it's
# covered by `test_raises_mixed_capture_list` instead.
CAPTURE_MODES = ["var", "var^", "read", "mut"]

# Type expressions accepted after `raises` in a closure. `Foo.Bar` requires a
# `struct Foo` prelude.
RAISES_TYPES = ["Error", "(Error)", "Foo.Bar"]


@pytest.mark.parametrize("mode", CAPTURE_MODES)
@pytest.mark.parametrize("raises", ["", " raises"])
def test_single_mode_capture(mode, raises):
    """Formats single-mode captures, with and without ``raises``."""
    source = (
        "def main() raises:\n"
        "    @always_inline\n"
        f"    def cb[](){raises} {{{mode}}}:\n"
        "        var x = 1\n"
    )
    assert_mojo_format(source, source)


@pytest.mark.parametrize("raises", ["", " raises"])
def test_empty_captures(raises):
    """Formats closures with empty captures."""
    source = (
        "def main() raises:\n"
        "    @always_inline\n"
        f"    def cb[](){raises} {{}}:\n"
        "        pass\n"
    )
    assert_mojo_format(source, source)


def test_raises_mixed_capture_list():
    """Formats a mixed capture list (``var``/``read``/``mut``/``ref``) with named captures."""
    source = (
        "def main() raises:\n"
        "    var a: Int = 0\n"
        "    var b: Int = 0\n"
        "    var c: Int = 0\n"
        "    var d: Int = 0\n"
        "\n"
        "    @always_inline\n"
        "    def cb[]() raises {var a, read b, mut c, ref d}:\n"
        "        var x = a + b + d\n"
        "        c = 1\n"
    )
    assert_mojo_format(source, source)


def test_var_move_named_capture():
    """Preserves ``var^ name`` (move marker on a named capture) tight."""
    source = (
        "def main() raises:\n"
        "    var a: Int = 0\n"
        "\n"
        "    @always_inline\n"
        "    def cb[]() raises {var^ a}:\n"
        "        var x = a\n"
    )
    assert_mojo_format(source, source)


def test_var_move_named_capture_trailing_caret():
    """Preserves ``var name^`` (legacy move-marker position) tight."""
    source = (
        "def main() raises:\n"
        "    var a: Int = 0\n"
        "\n"
        "    @always_inline\n"
        "    def cb[]() raises {var a^}:\n"
        "        var x = a\n"
    )
    assert_mojo_format(source, source)


@pytest.mark.parametrize("raises_type", RAISES_TYPES)
@pytest.mark.parametrize("space", ["", " "])
def test_raises_typed_exception_then_captures(raises_type, space):
    """Preserves the ``raises`` type expression unchanged before ``{var}``."""
    source = (
        "struct Foo:\n"
        "    alias Bar = Error\n"
        "\n"
        "\n"
        "def main() raises:\n"
        "    var y: Int = 0\n"
        "\n"
        "    @always_inline\n"
        f"    def cb[]() raises {raises_type}{space}{{var}}:\n"
        "        var x = y\n"
    )
    expected = (
        "struct Foo:\n"
        "    alias Bar = Error\n"
        "\n"
        "\n"
        "def main() raises:\n"
        "    var y: Int = 0\n"
        "\n"
        "    @always_inline\n"
        f"    def cb[]() raises {raises_type} {{var}}:\n"
        "        var x = y\n"
    )
    assert_mojo_format(source, expected)
