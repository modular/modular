# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import codecs
import os
import subprocess
import sys

from mblack import patched_main

if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
    os.chdir(directory)


def get_changed_files() -> list[str]:
    merge_base_result = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        capture_output=True,
    )
    merge_base = merge_base_result.stdout.decode().rstrip("\n")

    changed_files_result = subprocess.run(
        ["git", "diff", "--name-only"] + ([merge_base] if merge_base else []),
        capture_output=True,
    )
    changed_files_out = (
        codecs.escape_decode(changed_files_result.stdout)[0].decode().rstrip()  # type: ignore
    )
    changed_files = [
        line.lstrip('"').rstrip('"') for line in changed_files_out.splitlines()
    ]
    return [
        file
        for file in changed_files
        if file.endswith(".mojo") or file.endswith(".🔥")
    ]


if __name__ == "__main__":
    if os.getenv("FAST"):
        changed_files = get_changed_files()
        if not changed_files:
            # mblack errors if no paths are specified, so short circuit here
            exit(0)
        sys.argv = sys.argv + changed_files

    patched_main()
