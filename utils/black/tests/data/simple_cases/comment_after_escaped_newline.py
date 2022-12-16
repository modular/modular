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
#   Path:   tests/data/simple_cases/comment_after_escaped_newline.py
#
# ===----------------------------------------------------------------------=== #


def bob():  # pylint: disable=W9016
    pass


def bobtwo():  # some comment here
    pass


# output


def bob():  # pylint: disable=W9016
    pass


def bobtwo():  # some comment here
    pass
