# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Provides utilities for setting up linters

import os
import subprocess


def is_check() -> bool:
    # Default to format
    return os.getenv("CHECK", "0").lower() in {"1", "true"}


def is_fast() -> bool:
    # Default to fast
    return os.getenv("FAST", "1").lower() in {"1", "true"}


def _has_jj() -> bool:
    # Prefer jj when available: in a non-colocated jj workspace `git ls-files` /
    # `git diff` can't see the working copy. `jj root` only succeeds inside a jj
    # repo, so this gracefully falls through to git in plain git checkouts.
    return (
        subprocess.run(["bash", "-c", "command -v jj"]).returncode == 0
        and subprocess.run(["jj", "root"], capture_output=True).returncode == 0
    )


def _get_files(stdout: bytes) -> set[str]:
    return set(stdout.decode().splitlines())


def _submodule_paths() -> list[str]:
    # Submodule paths relative to the superproject root. `git ls-files` /
    # `git diff` treat a submodule as a single gitlink and don't descend into
    # it, so we run git inside each submodule separately (see below).
    result = subprocess.run(
        ["git", "submodule", "--quiet", "foreach", "echo $sm_path"],
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return result.stdout.decode().splitlines()


def _prefixed(prefix: str, files: set[str]) -> set[str]:
    return {f"{prefix}/{f}" for f in files}


def _git_all_files(cwd: str = ".") -> set[str]:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=cwd)
    deleted = subprocess.check_output(["git", "ls-files", "--deleted"], cwd=cwd)
    return _get_files(tracked) - _get_files(deleted)


def _git_changed_files(diff_target: str, cwd: str = ".") -> set[str]:
    result = subprocess.check_output(
        ["git", "diff", "--diff-filter=d", "--name-only", diff_target],
        cwd=cwd,
    )
    return _get_files(result)


def _submodule_diff_target(sm_path: str, super_target: str) -> str | None:
    # The submodule commit recorded in the superproject at `super_target`; this
    # is the right base to diff a submodule's working tree against. Returns None
    # if the submodule isn't a gitlink there or hasn't fetched that commit.
    rev = subprocess.run(
        ["git", "rev-parse", f"{super_target}:{sm_path}"], capture_output=True
    )
    if rev.returncode != 0:
        return None
    target = rev.stdout.decode().strip()
    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{target}^{{commit}}"],
        cwd=sm_path,
        capture_output=True,
    )
    return target if has_commit.returncode == 0 else None


def _oss_filter(files: set[str]) -> set[str]:
    # Only used for testing with the OSS overlay
    if not os.getenv("USE_OSS_FILTER"):
        return files

    return {
        f.removeprefix("oss/modular/")
        for f in files
        if f.startswith("oss/modular/")
    }


def get_all_files() -> set[str]:
    if _has_jj():
        return _oss_filter(
            _get_files(subprocess.check_output(["jj", "file", "list"]))
        )
    else:
        files = _git_all_files()
        for sm_path in _submodule_paths():
            files |= _prefixed(sm_path, _git_all_files(sm_path))
        return _oss_filter(files)


def get_changed_files() -> set[str]:
    if _has_jj():
        return _oss_filter(
            _get_files(
                subprocess.check_output(
                    [
                        "jj",
                        "diff",
                        "--name-only",
                        "--from",
                        "main@origin",
                        "--to",
                        "@",
                    ],
                )
            )
        )
    else:
        # TODO: Need a way to use a different diff base
        if lint_diff_target := os.getenv("LINT_DIFF_TARGET"):
            diff_target = lint_diff_target
        else:
            diff_target = (
                subprocess.check_output(
                    ["git", "merge-base", "origin/main", "HEAD"],
                )
                .decode()
                .rstrip("\n")
            )

        files = _git_changed_files(diff_target)
        for sm_path in _submodule_paths():
            sm_target = _submodule_diff_target(sm_path, diff_target)
            if sm_target is None:
                continue
            files |= _prefixed(
                sm_path, _git_changed_files(sm_target, cwd=sm_path)
            )
        return _oss_filter(files)
