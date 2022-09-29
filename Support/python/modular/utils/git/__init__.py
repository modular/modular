# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._git import fetch_checkout_commit, is_full_git_sha, shallow_clone

# Remove from the namespace so that it's not visible to users.
del _git  # noqa: F821
