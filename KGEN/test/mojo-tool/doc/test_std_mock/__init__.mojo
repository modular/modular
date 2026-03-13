# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Mock standard library package for testing stability tracking in mojo doc output."""


@stable
struct StableStruct:
    """A stable struct."""

    pass


struct UnstableStruct:
    """An unstable struct."""

    pass


@stable(since="1.0")
def stable_fn_with_version():
    """A stable function with a version string."""
    pass
