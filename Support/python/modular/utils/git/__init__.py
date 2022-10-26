# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._git import (
    GitError,
    branch_exists,
    check_gh_installed,
    fetch_checkout_commit,
    get_current_branch_name,
    get_gh_username,
    is_full_git_sha,
    shallow_clone,
)

# Remove from the namespace so that it's not visible to users.
del _git  # noqa: F821
