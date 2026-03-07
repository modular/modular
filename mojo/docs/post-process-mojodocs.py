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

import glob
import os
import re
import sys


def _is_heading(line: str, in_code_block: bool) -> re.Match[str] | None:
    """Match a markdown heading only when not inside a fenced code block."""
    if in_code_block:
        return None
    return re.match(r"^(#+)\s", line)


def _remove_empty_headings(lines: list[str]) -> list[str]:
    """Remove headings whose sections have no content.

    A heading is considered empty when the next non-blank line is another
    heading at the same or higher level, or there is no following content.
    A heading is kept if the next non-blank line is a deeper subsection.
    """
    in_code_block = False
    result: list[str] = []
    eat_blank = False
    for i, line in enumerate(lines):
        if eat_blank:
            eat_blank = False
            if not line.strip():
                continue
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        heading_match = _is_heading(line, in_code_block)
        if heading_match:
            level = len(heading_match.group(1))
            lookahead_code_block = in_code_block
            next_text = None
            for following in lines[i + 1 :]:
                if following.strip().startswith("```"):
                    lookahead_code_block = not lookahead_code_block
                if following.strip():
                    next_text = (following, lookahead_code_block)
                    break
            if next_text is None:
                eat_blank = True
                continue
            next_line, next_in_code = next_text
            next_heading = _is_heading(next_line, next_in_code)
            if next_heading:
                next_level = len(next_heading.group(1))
                if next_level <= level:
                    eat_blank = True
                    continue
        result.append(line)
    return result


def _parse_version(filename: str) -> tuple[int, ...] | None:
    """Parse a version tuple from a filename like 'v0.26.1.md'."""
    m = re.match(r"^v([\d.]+)\.md$", filename)
    if m:
        return tuple(int(x) for x in m.group(1).split("."))
    return None


def demote_all_headings(file_path) -> None:  # noqa: ANN001
    with open(file_path, "r+") as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            if not in_code_block and line.strip().startswith("#"):
                file.write("#" + line)
            else:
                file.write(line)


def assemble_mojo_changelog(base_path: str) -> None:
    """Build changelog/index.md from nightly-changelog.md and per-version files."""
    changelog_dir = os.path.join(base_path, "changelog")
    index_path = os.path.join(changelog_dir, "index.md")
    nightly_path = os.path.join(base_path, "nightly-changelog.md")

    with open(index_path) as f:
        index_content = f.read()

    with open(nightly_path) as f:
        nightly_lines = f.readlines()
    nightly_lines = _remove_empty_headings(nightly_lines)

    candidates = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(changelog_dir, "v*.md"))
    ]
    version_files = []
    for fname in candidates:
        if _parse_version(fname) is None:
            print(
                f"WARNING: skipping '{fname}' in changelog/ "
                f"(filename does not match vX.Y.Z.md pattern)"
            )
        else:
            version_files.append(fname)
    version_files.sort(key=lambda f: _parse_version(f) or (), reverse=True)

    # Assemble in chronological order: frontmatter/intro, then nightly
    # (unreleased), then released versions newest-first, then archive.
    assembled = index_content.rstrip() + "\n\n"
    assembled += "".join(nightly_lines).rstrip() + "\n"
    for vfile in version_files:
        with open(os.path.join(changelog_dir, vfile)) as f:
            assembled += "\n" + f.read().rstrip() + "\n"

    archive_path = os.path.join(changelog_dir, "archive.md")
    if os.path.exists(archive_path):
        with open(archive_path) as f:
            assembled += "\n" + f.read().rstrip() + "\n"

    with open(index_path, "w") as f:
        f.write(assembled)

    demote_all_headings(index_path)


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
    assemble_mojo_changelog(sys.argv[1])
    # TODO: Delete the following once we launch the new Mojo website (and fix the links)
    strip_mojo_path_prefix(sys.argv[1])
