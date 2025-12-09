# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""GitHub Actions utilities."""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def github_log_group(title: str) -> Generator[None, None, None]:
    """
    Context manager that creates a collapsible log group in GitHub Actions.

    The group is collapsed by default, hiding the enclosed output unless
    the user expands it in the GitHub Actions log viewer.

    Args:
        title: The title of the group.
    """
    print(f"::group::{title}", flush=True)
    try:
        yield
    finally:
        print("::endgroup::", flush=True)
