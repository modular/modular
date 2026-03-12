# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for f-strings and t-strings with nested quotes (same quote character)."""

import pytest
from mblack import format_str, Mode, TargetVersion


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_double_quote_nested(prefix):
    """Test nested double quotes in double-quoted string."""
    source = f'{prefix}"hello {{"world"}}"'
    expected = f'{prefix}"hello {{"world"}}"\n'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_single_quote_nested(prefix):
    """Test nested single quotes in single-quoted string."""
    source = f"{prefix}'hello {{'world'}}'"  # noqa: F541
    expected = f"{prefix}'hello {{'world'}}'\n"
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_multiple_interpolations(prefix):
    """Test multiple nested interpolations."""
    source = f'{prefix}"a {{"b"}} c {{"d"}}"'
    expected = f'{prefix}"a {{"b"}} c {{"d"}}"\n'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_triple_quoted(prefix):
    """Test triple-quoted strings with nested quotes."""
    source = f'{prefix}"""hello {{"world"}}"""'
    expected = f'{prefix}"""hello {{"world"}}"""\n'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_complex_expression(prefix):
    """Test complex expression with function calls."""
    source = f'{prefix}"result: {{foo("bar", "baz")}}"'
    expected = f'{prefix}"result: {{foo("bar", "baz")}}"\n'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_escaped_braces(prefix):
    """Test escaped braces alongside interpolations."""
    source = prefix + '"data: {{key: {"value"}}}"'
    expected = prefix + '"data: {{key: {"value"}}}"\n'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_multiline(prefix):
    """Test multiline triple-quoted strings."""
    source = f'''{prefix}"""
hello {{"world"}}
and {{"universe"}}
"""'''
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert "hello" in result and "world" in result


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_nested_different_quotes(prefix):
    """Test nested strings using different quote characters."""
    source = f'''{prefix}"hello {{'world'}}"'''
    expected = f'''{prefix}"hello {{'world'}}"\n'''
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert result == expected


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_deeply_nested(prefix):
    """Test deeply nested same-quote strings."""
    source = prefix + '"a {' + prefix + '"b {' + prefix + '"c"}"}"'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert "a" in result and "b" in result and "c" in result


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_malformed_input_raises(prefix):
    """Test that malformed input raises an exception."""
    source = f'{prefix}"hello {{world"'  # Unclosed brace
    with pytest.raises(Exception):
        format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))


def test_fstring_quote_normalization_skipped():
    """F-strings skip quote normalization to preserve interpolations."""
    source = r'f"test {"say \"hello\""}"'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    expected = 'f"test {"say \\"hello\\""}"\n'
    assert result == expected


def test_mixed_fstring_and_tstring():
    """Test f-strings and t-strings can coexist."""
    source = 'x = f"hello {name}" + t"world {value}"'
    result = format_str(source, mode=Mode(target_versions={TargetVersion.MOJO}))
    assert "f" in result and "t" in result


MOJO_MODE = Mode(target_versions={TargetVersion.MOJO})


@pytest.mark.parametrize(
    "prefix",
    ["rt", "rT", "Rt", "RT", "tr", "tR", "Tr", "TR"],
)
def test_raw_tstring_prefix_normalized_to_rt(prefix):
    """All raw t-string prefix variants are normalized to 'rt'."""
    source = f'{prefix}"hello {{name}}"'
    result = format_str(source, mode=MOJO_MODE)
    assert result == 'rt"hello {name}"\n'


@pytest.mark.parametrize(
    "prefix",
    ["rf", "rF", "Rf", "RF", "fr", "fR", "Fr", "FR"],
)
def test_raw_fstring_prefix_normalized_to_rf(prefix):
    """All raw f-string prefix variants are normalized to 'rf'."""
    source = f'{prefix}"hello {{name}}"'
    result = format_str(source, mode=MOJO_MODE)
    assert result == 'rf"hello {name}"\n'
