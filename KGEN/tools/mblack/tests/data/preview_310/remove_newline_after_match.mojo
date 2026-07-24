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
#   Path:   tests/data/preview_310/remove_newline_after_match.py
#
# ===----------------------------------------------------------------------=== #


def http_status(status):

    match status:

        case 400:

            return "Bad request"

        case 401:

            return "Unauthorized"

        case 403:

            return "Forbidden"

        case 404:

            return "Not found"


# output
def http_status(status):
    match status:
        case 400:
            return "Bad request"

        case 401:
            return "Unauthorized"

        case 403:
            return "Forbidden"

        case 404:
            return "Not found"
