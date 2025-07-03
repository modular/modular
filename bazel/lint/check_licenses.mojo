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

import sys
from pathlib import Path, cwd
from os import listdir
from collections import Set
from subprocess import run

# We can't check much more than this at the moment, because the license year
# changes and the language is not mature enough to do regex yet.
alias LICENSE = """# ===----------------------------------------------------------------------=== #
# Copyright (c)"""


fn is_ignored_file(filename: StringSlice) -> Bool:
    if not (
        filename.endswith(".py")
        or filename.endswith(".mojo")
        or filename.endswith(".🔥")
    ):
        return True

    # Generated files
    if (
        filename == "max/serve/schemas/kserve.py"
        or filename == "max/serve/schemas/openai.py"
    ):
        return True

    return False


fn get_git_files() raises -> Set[String]:
    # Need to get tracked, untracked, and deleted files separately
    tracked = run("git ls-files")
    untracked = run("git ls-files --exclude-standard --others")
    deleted = run("git ls-files --deleted")

    fn _get_files(stdout: String) -> Set[String]:
        result = Set[String]()
        for file in stdout.split("\n"):
            # Manually replace escaped 🔥 with a literal 🔥
            newfile = file.replace(r"\360\237\224\245", "🔥").strip('"')

            if not is_ignored_file(newfile):
                result.add(String(newfile))
        return result

    return (_get_files(tracked) | _get_files(untracked)) - _get_files(deleted)


fn check_path(path: Path, mut files_without_license: List[Path]) raises:
    file_text = path.read_text()

    # Ignore #! in scripts
    if file_text.startswith("#!"):
        has_license = "\n".join(file_text.splitlines()[1:]).startswith(LICENSE)
    else:
        has_license = file_text.startswith(LICENSE)

    if not has_license:
        files_without_license.append(path)


fn main() raises:
    target_paths = sys.argv()

    files_without_license = List[Path]()
    if len(target_paths) < 2:
        for file in get_git_files():
            check_path(file, files_without_license)
    else:
        for i in range(len(target_paths)):
            if i == 0:
                # this is the current file
                continue
            path = Path(target_paths[i])
            check_path(path, files_without_license)

    if len(files_without_license) > 0:
        print("The following files have missing licences 💥 💔 💥")
        for file in files_without_license:
            print(file)
        print("Please add the license to each file before committing.")
        sys.exit(1)
