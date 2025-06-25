# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import pytest
from modular.utils.git import (
    GitError,
    branch_exists,
    get_changed_dirs,
    get_current_branch_name,
    get_uncommitted_changes,
    is_full_git_sha,
    shallow_clone,
)
from modular.utils.subprocess import run_shell_command

# ===----------------------------------------------------------------------=== #
# Test utilities
# ===----------------------------------------------------------------------=== #


def make_dummy_repo(repo_dir: Path) -> None:
    # Helper to set up a repo for testing.
    run_shell_command(["git", "init", "-q"], cwd=repo_dir)
    run_shell_command(["git", "config", "user.name", "Tester"], cwd=repo_dir)
    run_shell_command(
        ["git", "config", "user.email", "ci@modular.com"], cwd=repo_dir
    )


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


def test_is_full_git_sha() -> None:
    assert is_full_git_sha("0123456789012345678901234567890123456789")
    assert is_full_git_sha("0b7349102db619105fb282c2340a64c44e4adbe6")
    assert not is_full_git_sha("0b73491")
    assert not is_full_git_sha("0B7349102DB619105FB282C2340A64C44E4ADBE6")
    assert not is_full_git_sha("acorrectlengthnothexstringthatshouldfail")


# Test with commit SHA and branch name.
@pytest.mark.parametrize(
    "ref", ["44679f67d234667eaeb138dbcde468669a5181a8", "main"]
)
def test_shallow_clone(tmp_path: Path, ref: str) -> None:
    url = "https://github.com/actions/checkout.git"
    shallow_clone(tmp_path, url, ref, remove_git=True)
    assert (tmp_path / "README.md").exists()
    assert not (tmp_path / ".git").exists()


def test_branch_exists(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        branch_exists("some/branch", repo_dir=tmp_path)

    make_dummy_repo(tmp_path)
    assert not branch_exists("some/branch", repo_dir=tmp_path)

    run_shell_command(["git", "checkout", "-b", "some/branch"], cwd=tmp_path)
    run_shell_command(
        ["git", "commit", "--allow-empty", "-m", "Some message"],
        cwd=tmp_path,
    )
    assert branch_exists("some/branch", repo_dir=tmp_path)


def test_get_current_branch_name(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        get_current_branch_name(repo_dir=tmp_path)

    make_dummy_repo(tmp_path)
    run_shell_command(["git", "checkout", "-b", "main"], cwd=tmp_path)
    with pytest.raises(GitError):
        get_current_branch_name(repo_dir=tmp_path)

    run_shell_command(
        ["git", "commit", "--allow-empty", "-m", "Some message"],
        cwd=tmp_path,
    )
    assert get_current_branch_name(repo_dir=tmp_path) == "main"


def test_get_uncommitted_changes(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        get_current_branch_name(repo_dir=tmp_path)

    make_dummy_repo(tmp_path)
    assert not get_uncommitted_changes(tmp_path)

    some_file = tmp_path / "some_file.txt"
    other_file = tmp_path / "other_file.txt"
    some_file.touch(exist_ok=False)
    other_file.touch(exist_ok=False)

    changes = get_uncommitted_changes(tmp_path)
    assert len(changes) == 1
    assert changes["??"] == ["other_file.txt", "some_file.txt"]

    run_shell_command(["git", "add", some_file, other_file], cwd=tmp_path)
    changes = get_uncommitted_changes(tmp_path)
    assert len(changes) == 1
    assert changes["A "] == ["other_file.txt", "some_file.txt"]

    with open(some_file, "w") as f:
        f.write("This is some file")

    changes = get_uncommitted_changes(tmp_path)
    assert len(changes) == 2
    assert changes["A "] == ["other_file.txt"]
    assert changes["AM"] == ["some_file.txt"]


def test_get_changed_dirs(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        get_current_branch_name(repo_dir=tmp_path)

    make_dummy_repo(tmp_path)
    assert not get_uncommitted_changes(tmp_path)

    some_file = tmp_path / "dir1" / "some_file.txt"
    other_file = tmp_path / "dir2" / "other_file.txt"
    some_file.parent.mkdir(parents=True)
    some_file.touch(exist_ok=False)
    other_file.parent.mkdir(parents=True)
    other_file.touch(exist_ok=False)

    # Make a commit in the dummy repo
    run_shell_command(["git", "add", some_file, other_file], cwd=tmp_path)
    run_shell_command(["git", "commit", "-m", "fake"], cwd=tmp_path)

    with open(other_file, "w") as f:
        f.write("hello")

    # Make another commit so we can compare.
    run_shell_command(["git", "add", other_file], cwd=tmp_path)
    run_shell_command(["git", "commit", "-m", "fake2"], cwd=tmp_path)

    changed_dirs = get_changed_dirs("HEAD~1", tmp_path)
    assert len(changed_dirs) == 1
    assert Path("dir2") in changed_dirs
