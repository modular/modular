# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import shutil
from pathlib import Path

from modular.utils.git import is_full_git_sha, shallow_clone


def test_is_full_git_sha():
    assert is_full_git_sha("0123456789012345678901234567890123456789")
    assert is_full_git_sha("0b7349102db619105fb282c2340a64c44e4adbe6")
    assert not is_full_git_sha("0b73491")
    assert not is_full_git_sha("0B7349102DB619105FB282C2340A64C44E4ADBE6")
    assert not is_full_git_sha("acorrectlengthnothexstringthatshouldfail")


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
