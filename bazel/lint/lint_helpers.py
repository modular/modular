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


def get_all_files() -> set[str]:
    if _has_jj():
        return _get_files(subprocess.check_output(["jj", "file", "list"]))
    else:
        tracked = subprocess.check_output(["git", "ls-files"])
        deleted = subprocess.check_output(["git", "ls-files", "--deleted"])

        return _get_files(tracked) - _get_files(deleted)


def get_changed_files() -> set[str]:
    if _has_jj():
        return _get_files(
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
    else:
        merge_base_result = subprocess.check_output(
            ["git", "merge-base", "origin/main", "HEAD"],
        )
        merge_base = merge_base_result.decode().rstrip("\n")

        changed_files_result = subprocess.check_output(
            ["git", "diff", "--diff-filter=d", "--name-only", merge_base]
        )
        return _get_files(changed_files_result)
