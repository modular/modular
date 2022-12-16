# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/preview/docstring_preview.py
#
# ===----------------------------------------------------------------------=== #


def docstring_almost_at_line_limit():
    """long docstring................................................................."""


def docstring_almost_at_line_limit_with_prefix():
    f"""long docstring................................................................"""


def mulitline_docstring_almost_at_line_limit():
    """long docstring.................................................................

    ..................................................................................
    """


def mulitline_docstring_almost_at_line_limit_with_prefix():
    f"""long docstring................................................................

    ..................................................................................
    """


def docstring_at_line_limit():
    """long docstring................................................................"""


def docstring_at_line_limit_with_prefix():
    f"""long docstring..............................................................."""


def multiline_docstring_at_line_limit():
    """first line-----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def multiline_docstring_at_line_limit_with_prefix():
    f"""first line----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def single_quote_docstring_over_line_limit():
    "We do not want to put the closing quote on a new line as that is invalid (see GH-3141)."


def single_quote_docstring_over_line_limit2():
    "We do not want to put the closing quote on a new line as that is invalid (see GH-3141)."


# output


def docstring_almost_at_line_limit():
    """long docstring................................................................."""


def docstring_almost_at_line_limit_with_prefix():
    f"""long docstring................................................................"""


def mulitline_docstring_almost_at_line_limit():
    """long docstring.................................................................

    ..................................................................................
    """


def mulitline_docstring_almost_at_line_limit_with_prefix():
    f"""long docstring................................................................

    ..................................................................................
    """


def docstring_at_line_limit():
    """long docstring................................................................"""


def docstring_at_line_limit_with_prefix():
    f"""long docstring..............................................................."""


def multiline_docstring_at_line_limit():
    """first line-----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def multiline_docstring_at_line_limit_with_prefix():
    f"""first line----------------------------------------------------------------------

    second line----------------------------------------------------------------------"""


def single_quote_docstring_over_line_limit():
    "We do not want to put the closing quote on a new line as that is invalid (see GH-3141)."


def single_quote_docstring_over_line_limit2():
    "We do not want to put the closing quote on a new line as that is invalid (see GH-3141)."
