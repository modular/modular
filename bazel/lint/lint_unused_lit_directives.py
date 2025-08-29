#!/usr/bin/env python3
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

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_LIT_REGEX = re.compile(
    r"\b(RUN|REQUIRES|UNSUPPORTED|XFAIL|CHECK(?:-[^ :]+)?):", re.MULTILINE
)

_TEST_FILES_QUERY = """
let tests = tests(//...) - attr(tags, clang-tidy, //...) in
let test_source_files = kind("source file", deps($tests, 1)) in
$test_source_files intersect //...:*
"""

_LIT_TEST_FILES_QUERY = """
let tests = attr(tags, lit, tests(//...)) in
let test_source_files = kind("source file", deps($tests, 1)) in
$test_source_files intersect //...:*
"""

_FILECHECK_TEST_FILES_QUERY = """
let tests = attr(tags, filecheck, tests(//...)) in
let test_source_files = kind("source file", deps($tests, 1)) in
$test_source_files intersect //...:*
"""


def _bazelw() -> Path:
    return (
        Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        )
        / "bazelw"
    )


def _label_to_path(label: str) -> Path:
    assert label.startswith("//")
    return Path(label[2:].replace(":", "/"))


def _get_files(bazel: Path, query: str) -> set[Path]:
    return {
        _label_to_path(x)
        for x in (
            subprocess.check_output(
                [bazel, "query", query],
                text=True,
            )
            .strip()
            .splitlines()
        )
    }


def _get_lit_directives(file: Path) -> set[str]:
    with file.open() as f:
        return set(_LIT_REGEX.findall(f.read()))


def _main() -> None:
    bazel = _bazelw()
    all_test_files = _get_files(bazel, _TEST_FILES_QUERY)
    lit_test_files = _get_files(bazel, _LIT_TEST_FILES_QUERY)
    filecheck_test_files = _get_files(bazel, _FILECHECK_TEST_FILES_QUERY)

    if not all_test_files or not lit_test_files or not filecheck_test_files:
        print(
            "error: failed to query test files, likely an error in the linter."
        )
        exit(1)

    errors = []
    for file in all_test_files - lit_test_files:
        directive = _get_lit_directives(file)
        if directive:
            filecheck_only = all(x.startswith("CHECK") for x in directive)
            if filecheck_only:
                if file not in filecheck_test_files:
                    errors.append(
                        f"error: {file}: has a CHECK line but is not a filecheck test file, either change it to use the 'mojo_filecheck_test' or 'lit_tests' rules, or remove the CHECK line"
                    )
            else:
                errors.append(
                    f"error: {file}: has a RUN line but is not a lit test file, either change it to use the 'lit_tests' rule, or remove the RUN lines"
                )

    if errors:
        print("\n".join(sorted(errors)))
        exit(1)


if __name__ == "__main__":
    _main()
