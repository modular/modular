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
#   Path:   tests/conftest.py
#
# ===----------------------------------------------------------------------=== #

from tests import util

pytest_plugins = ["tests.optional"]


def pytest_addoption(parser):
    parser.addoption(
        "--validate-with-mojo-build",
        action="store_true",
        default=False,
        help=(
            "Also compile every sample passed to assert_mojo_format with "
            "`mojo build` to verify it is valid Mojo."
        ),
    )


def pytest_configure(config):
    if config.getoption("--validate-with-mojo-build"):
        util.VALIDATE_WITH_MOJO_BUILD = True
