# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def pytest_addoption(parser):
    parser.addoption(
        "--custom-ops-path",
        type=str,
        default="",
        help="Path to custom Ops package",
    )
