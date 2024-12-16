# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Git and github (gh) utilities

This module collects commonly used functionality for writing scripts that deal
with git repositories and the github API. It is not (yet) a goal of this module
to hide all git-related functionality behind a stable API. For now, users are
just encouraged to factor out snippets that they think might be useful for
others, or could benefit from more rigorous testing and type hinting (which is,
admittedly, most Python code not meant to be thrown away immediately).
"""

from ._git import (
    GitError,
    branch_exists,
    check_gh_installed,
    fetch_checkout_commit,
    get_changed_dirs,
    get_changed_files,
    get_current_branch_name,
    get_gh_username,
    get_uncommitted_changes,
    is_full_git_sha,
    shallow_clone,
)

# Remove from the namespace so that it's not visible to users.
del _git  # type: ignore # noqa: F821
