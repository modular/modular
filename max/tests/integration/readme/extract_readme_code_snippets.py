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
"""Pulls the shell snippets out of the top-level README for the tests here.

Every ``bash`` block in the named section must be a single
``./bazelw run <label> -- <args>`` command. Bazel cannot be invoked from inside
a Bazel test, so each test maps the label onto a binary it already depends on
and runs that with the snippet's arguments.
"""

import re
from pathlib import Path

BAZEL_RUN = re.compile(r"\./bazelw run (//\S+) -- (.*)")

# Every label a snippet may run, and the test that runs it. A snippet using a
# label missing here fails in every test rather than being skipped by all of
# them.
COVERED_LABELS = {
    "//Mojo:mojo": "test_readme_mojo_snippet",
    "//max/python/max/_entrypoints:pipelines": "test_readme_max_snippet",
}


def bash_commands(readme: Path, section: str) -> list[str]:
    """Returns the commands in the bash blocks under one ``##`` heading."""
    text = readme.read_text()
    heading = f"\n## {section}\n"
    assert heading in text, f"no section '## {section}' in {readme}"
    body = text.split(heading, 1)[1].split("\n## ", 1)[0]
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", body, re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            if line.strip():
                commands.append(" ".join(line.split()))
    assert commands, f"no bash snippets under '## {section}' in {readme}"
    return commands


def bazel_run_target(command: str) -> tuple[str, str]:
    """Splits a ``./bazelw run <label> -- <args>`` command into its parts."""
    match = BAZEL_RUN.fullmatch(command)
    assert match, f"expected `./bazelw run <label> -- <args>`, got: {command}"
    label, args = match.groups()
    assert label in COVERED_LABELS, (
        f"no test runs {label}; add one and list it in COVERED_LABELS"
    )
    return label, args
