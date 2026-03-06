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

"""Post-processing for mojo-only docs (mojosite build)."""

import os
import re
import sys


def copy_nightly_changelog(source, destination) -> None:  # noqa: ANN001
    with open(source) as source_file:
        source_lines = source_file.readlines()

    # Copy everything after UNRELEASED
    unreleased_index = next(
        (
            i
            for i, line in enumerate(source_lines)
            if re.match(r"^## UNRELEASED\b", line.strip())
        ),
        None,
    )
    if unreleased_index is None:
        raise ValueError("## UNRELEASED section not found in source file")
    copied_lines = source_lines[unreleased_index + 1 :]

    # Remove empty heading sections
    result_lines = []
    skip_next = False
    for i, line in enumerate(copied_lines):
        if skip_next:
            skip_next = False
            continue
        if re.match(r"^#{2,}\s", line):
            next_text_line = next(
                (l for l in copied_lines[i + 1 :] if l.strip()), None
            )
            if (
                next_text_line
                and re.match(r"^###\s", line)
                and re.match(r"^####\s", next_text_line)
            ):
                # Allow empty H3 followed by H4
                pass
            elif (
                next_text_line and re.match(r"^#{2,}\s", next_text_line)
            ) or next_text_line is None:
                skip_next = True
                continue
        result_lines.append(line)

    # Insert copied lines at INSERT HERE
    with open(destination) as dest_file:
        dest_lines = dest_file.readlines()

    updated_dest_lines = []
    inserted = False
    for line in dest_lines:
        if not inserted and "INSERT HERE" in line:
            updated_dest_lines.extend(result_lines)
            inserted = True
        else:
            updated_dest_lines.append(line)

    with open(destination, "w") as dest_file:
        dest_file.writelines(updated_dest_lines)

    # Delete the "nightly" source file after copying its contents
    try:
        os.remove(source)
    except OSError as e:
        print(f"Error deleting file {source}: {e}")


def replace_relative_paths(docs_path, file_path) -> None:  # noqa: ANN001
    path = docs_path + file_path
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    content = f.read()
                # Replace relative link paths "](./"
                new_path = root.replace(docs_path, "")
                new_content = re.sub(r"\]\(\./", f"]({new_path}/", content)
                with open(file_path, "w") as f:
                    f.write(new_content)


def remove_docs_domain(file_path) -> None:  # noqa: ANN001
    for root, _, files in os.walk(file_path):
        for filename in files:
            if filename.endswith((".md", ".ipynb")):
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as file:
                    content = file.read()
                    updated_content = re.sub(
                        r"https://docs.modular.com/", "/", content
                    )
                    file.seek(0)
                    file.write(updated_content)
                    file.truncate()


def strip_mojo_path_prefix(docs_path: str) -> None:
    """Remove '/mojo' prefix from hyperlinks in /manual and /tools."""
    for subdir in ("manual", "tools"):
        dir_path = os.path.join(docs_path, subdir)
        if not os.path.isdir(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if not filename.endswith((".md", ".mdx", ".ipynb")):
                    continue
                file_path = os.path.join(root, filename)
                with open(file_path, "r+") as f:
                    content = f.read()
                    updated = re.sub(
                        r"(\]\(|href=[\"'])/mojo/", r"\1/", content
                    )
                    if updated != content:
                        f.seek(0)
                        f.write(updated)
                        f.truncate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 post-process-mojodocs.py <directory>")
        sys.exit(1)

    replace_relative_paths(sys.argv[1], "/stdlib")
    remove_docs_domain(sys.argv[1])
    copy_nightly_changelog(
        sys.argv[1] + "/_nightly-changelog.md",
        sys.argv[1] + "/changelog.md",
    )
    # TODO: Delete the following once we launch the new Mojo website (and fix the links)
    strip_mojo_path_prefix(sys.argv[1])
