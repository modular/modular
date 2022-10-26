# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import shutil
import tempfile
from pathlib import Path

import pytest

from modular.utils.git import (
    GitError,
    branch_exists,
    get_current_branch_name,
    is_full_git_sha,
    shallow_clone,
)
from modular.utils.subprocess import run_shell_command


def test_is_full_git_sha():
    assert is_full_git_sha("0123456789012345678901234567890123456789")
    assert is_full_git_sha("0b7349102db619105fb282c2340a64c44e4adbe6")
    assert not is_full_git_sha("0b73491")
    assert not is_full_git_sha("0B7349102DB619105FB282C2340A64C44E4ADBE6")
    assert not is_full_git_sha("acorrectlengthnothexstringthatshouldfail")


# This is not an actual test, just a helper to reduce duplicates
def _test_shallow_clone(ref: str):
    artifact_root = Path(__file__).parent / ".artifacts" / "shallow_clone"
    url = "https://github.com/actions/checkout.git"

    clone_dir = artifact_root / "fetch"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    shallow_clone(clone_dir, url, ref, remove_git=True)
    assert (clone_dir / "README.md").exists()
    assert not (clone_dir / ".git").exists()


def test_shallow_clone_sha():
    _test_shallow_clone("44679f67d234667eaeb138dbcde468669a5181a8")


def test_shallow_clone_main():
    _test_shallow_clone("main")


def make_dummy_repo(repo_dir: Path):
    # Helper to set up a repo for testing
    run_shell_command(["git", "init", "-q"], cwd=repo_dir)
    run_shell_command(["git", "config", "user.name", "Tester"], cwd=repo_dir)
    run_shell_command(
        ["git", "config", "user.email", "ci@modular.com"], cwd=repo_dir
    )


def test_branch_exists():
    # we have to test this outside our monorepo
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dir = Path(tmp_dir)

        with pytest.raises(GitError):
            branch_exists("some/branch", repo_dir=dummy_dir)

        make_dummy_repo(dummy_dir)
        assert not branch_exists("some/branch", repo_dir=dummy_dir)

        run_shell_command(
            ["git", "checkout", "-b", "some/branch"], cwd=dummy_dir
        )
        run_shell_command(
            ["git", "commit", "--allow-empty", "-m", "Some message"],
            cwd=dummy_dir,
        )
        assert branch_exists("some/branch", repo_dir=dummy_dir)


def test_get_current_branch_name():
    # we have to test this outside our monorepo
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_dir = Path(tmp_dir)

        with pytest.raises(GitError):
            get_current_branch_name(repo_dir=dummy_dir)

        make_dummy_repo(dummy_dir)
        run_shell_command(["git", "checkout", "-b", "main"], cwd=dummy_dir)
        with pytest.raises(GitError):
            get_current_branch_name(repo_dir=dummy_dir)

        run_shell_command(
            ["git", "commit", "--allow-empty", "-m", "Some message"],
            cwd=dummy_dir,
        )
        assert get_current_branch_name(repo_dir=dummy_dir) == "main"
