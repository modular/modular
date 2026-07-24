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
#   Path:   tests/data/simple_cases/string_prefixes.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3

name = "Łukasz"
(f"hello {name}", f"hello {name}")
(b"", b"")
("", "")
(r"", R"")

(rf"", rf"", Rf"", Rf"", rf"", rf"", Rf"", Rf"")
(rb"", rb"", Rb"", Rb"", rb"", rb"", Rb"", Rb"")


def docstring_singleline():
    R"""2020 was one hell of a year. The good news is that we were able to"""


def docstring_multiline():
    R"""
    clear out all of the issues opened in that time :p
    """


# output


#!/usr/bin/env python3

name = "Łukasz"
(f"hello {name}", f"hello {name}")
(b"", b"")
("", "")
(r"", r"")

(rf"", rf"", rf"", rf"", rf"", rf"", rf"", rf"")
(rb"", rb"", rb"", rb"", rb"", rb"", rb"", rb"")


def docstring_singleline():
    r"""2020 was one hell of a year. The good news is that we were able to"""


def docstring_multiline():
    r"""
    clear out all of the issues opened in that time :p
    """
