# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import re
import shutil
import string
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

from modular.utils.subprocess import (
    CalledProcessError,
    get_command_output,
    run_shell_command,
)


class GitError(Exception):
    pass


def fetch_checkout_commit(repo_dir: Path, ref: str, remote: str = "origin"):
    """Helper function to quickly fetch and checkout a new ref.

    Args:
        repo_dir: path to an existing git repository.
        ref: a tag, brach, or (full) commit SHA.
        remote: git remote to use. Default: "origin".
    """

    run_shell_command(["git", "fetch", "--depth=1", remote, ref], cwd=repo_dir)
    run_shell_command(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)


def is_full_git_sha(s: str) -> bool:
    """Return True if the given string is a valid full git SHA.

    The string needs to consist of 40 lowercase hex characters.

    """
    if len(s) != 40:
        return False

    digits = set(string.hexdigits.lower())
    return all(c in digits for c in s)


def shallow_clone(
    clone_dir: Path, url: str, ref: str, remove_git: bool = False
):
    """Clone the given repo without any git history.

    This makes the cloning faster for repos with large histories.

    Args:
        clone_dir: path to the new clone directory. It is created if it doesn't
            already exist.
        url: repository url to clone from.
        ref: a tag, brach, or (full) commit SHA.
        remove_git: remove the .git directory after cloning.

    Raises:
        FileExistsError: if clone_dir exists and is not an empty directory.
    """

    if clone_dir.exists():
        if not clone_dir.is_dir() or any(clone_dir.iterdir()):
            raise FileExistsError(
                f"Clone directory already exists and is not empty: {clone_dir}"
            )
    else:
        clone_dir.mkdir(parents=True)

    run_shell_command(["git", "init"], cwd=clone_dir)
    run_shell_command(["git", "remote", "add", "origin", url], cwd=clone_dir)
    fetch_checkout_commit(clone_dir, ref)

    if remove_git:
        shutil.rmtree(clone_dir / ".git")


def branch_exists(branch: str, repo_dir: Optional[Path] = None) -> bool:
    """Returns whether a branch with the given name exists.

    Args:
        branch: branch name as a string.
        repo_dir: path to the repo. Defaults to the current working directory.

    Returns:
        True if the branch exists, False otherwise.

    Raises:
        GitError: if called outside a git repo.
    """
    proc = run_shell_command(
        ["git", "show-ref", "-q", f"refs/heads/{branch}"],
        stderr=subprocess.DEVNULL,
        cwd=repo_dir,
        check=False,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise GitError("Not inside a valid git repository.")


def get_current_branch_name(repo_dir: Optional[Path] = None) -> str:
    """Returns the name of the branch currently checked out.

    Args:
        repo_dir: path to the repo. Defaults to the current working directory.

    Returns:
        The name of the branch currently checked out, or "HEAD" if the repo is
        in a 'detached HEAD' state

    Raises:
        GitError: if called outside a git repo, or the repo doesn't have any
        commits yet.
    """

    try:
        return get_command_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir
        ).strip()
    except CalledProcessError as e:
        if e.returncode == 128:
            raise GitError("Not inside a valid git repository.") from None
        raise


def get_uncommitted_changes(
    repo_dir: Optional[Path] = None,
) -> Dict[str, Sequence[str]]:
    """Return a dictionary of uncommitted changes.

    Args:
        repo_dir: path to the repo. Defaults to the current working directory.

    Returns:
        A dictionary with keys as described in
        https://git-scm.com/docs/git-status#_short_format and values as lists
        of the corresponding changes, each change either in the format "PATH",
        or "ORIG_PATH -> PATH".

    Raises:
        GitError: if called outside a git repo.
    """
    try:
        out = get_command_output(["git", "status", "--porcelain"], cwd=repo_dir)
    except CalledProcessError as e:
        if e.returncode == 128:
            raise GitError("Not inside a valid git repository.") from None
        raise

    changes = {}
    for line in out.splitlines():
        # First two chars are the status, changed path starts at 4th character.
        changes.setdefault(line[:2], []).append(line[3:])
    return changes


# TODO: enforce this as a module dependency
def check_gh_installed():
    """Check if the gh tool is installed.

    Raises:
        GitError if gh is not available.
    """

    try:
        run_shell_command(["gh"], capture_output=True)
    except CalledProcessError:
        raise GitError(
            "'gh' is not installed. Please visit https://cli.github.com/ for"
            " installation instuctions."
        )


# TODO: figure out how to test this
def get_gh_username() -> str:
    """Return the current github username.

    Returns:
        Current github username as a string.

    Raises:
        GitError: if called outside a git repo, or.
    """

    user_query = get_command_output(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            "owner=UserCurrent",
            "-f",
            "query=query{viewer{login}}",
        ]
    )

    # Extract the login name.
    m = re.search(r"\"login\":\"(.*?)\"", user_query)
    if not m:
        raise GitError("Unable to find current github user name")

    return m.group(1)


def get_changed_files(
    base: Optional[str] = None, repo_dir: Optional[Path] = None
) -> Sequence[Path]:
    """Get the list of files changed between this commit and the base commit.

    Returns:
        A list of Path objects that correspond to the changed files.
    """
    get_file_changes = [
        "git",
        "diff",
        "--name-only",
        base if base is not None else "main",
        "HEAD",
    ]
    result = get_command_output(get_file_changes, cwd=repo_dir)
    return [Path(r) for r in result.split("\n")]


def get_changed_dirs(
    base: Optional[str] = None, repo_dir: Optional[Path] = None
) -> Set[Path]:
    """Get the list of top-level directories changed between this commit
       and the base commit.

    Returns:
        A list of Path objects that correspond to the directories that have
        files changed.
    """
    return {Path(file.parts[0]) for file in get_changed_files(base, repo_dir)}
