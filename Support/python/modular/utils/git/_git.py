import shutil
import string
from pathlib import Path

from modular.utils.subprocess import run_chained_commands


def fetch_checkout_commit(repo_dir: Path, ref: str, remote: str = "origin"):
    """Helper function to quickly fetch and checkout a new ref.

    Args:
        repo_dir: path to an existing git repository.
        ref: a tag, brach, or (full) commit SHA.
        remote: git remote to use. Default: "origin".
    """

    run_chained_commands(
        (
            ["git", "fetch", "--depth=1", remote, ref],
            ["git", "checkout", "FETCH_HEAD"],
        ),
        cwd=repo_dir,
    )


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

    run_chained_commands(
        (
            ["git", "init"],
            ["git", "remote", "add", "origin", url],
        ),
        cwd=clone_dir,
    )
    fetch_checkout_commit(clone_dir, ref)

    if remove_git:
        shutil.rmtree(clone_dir / ".git")
