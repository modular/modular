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
        # After removing a heading, eat one trailing blank line to avoid
        # leaving a double-blank gap.
        if eat_blank:
            eat_blank = False
            if not line.strip():
                continue
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        heading_match = _is_heading(line, in_code_block)
        if heading_match:
            level = len(heading_match.group(1))
            # Track code-block state separately so lookahead doesn't
            # mutate the outer in_code_block.
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


def _frontmatter_to_heading(text: str, edit_title: bool = True) -> str:
    """Replace YAML frontmatter with an H1 heading reconstructed from its fields.

    Converts frontmatter like:
        ---
        title: Mojo v0.26.2
        date: 2026-03-19
        ---
    into:
        # v0.26.2 (2026-03-19)

    When edit_title is False, the title is used as-is without stripping
    the "Mojo" prefix or appending the date.
    """
    m = re.match(
        r"^---\s*\n(.*?\n)---\s*\n",
        text,
        re.DOTALL,
    )
    if not m:
        return text
    front = m.group(1)
    title = ""
    date = ""
    for line in front.splitlines():
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("date:"):
            date = line.split(":", 1)[1].strip()
    if edit_title:
        title = re.sub(r"^Mojo\s+", "", title)
        heading = f"# {title} ({date})\n" if date else f"# {title}\n"
    else:
        heading = f"# {title}\n\n"
    return heading + text[m.end() :]


def assemble_changelog(base_path: str) -> None:
    """Assemble changelog/index.md from nightly-changelog.md and per-version files."""
    changelog_dir = os.path.join(base_path, "changelog")
    index_path = os.path.join(changelog_dir, "index.md")
    nightly_path = os.path.join(base_path, "nightly-changelog.md")

    with open(index_path) as f:
        index_content = f.read()

    # Include unreleased (nightly) entries only when GITHUB_BRANCH is
    # unset or "main".  Stable/release builds pass
    # --action_env=GITHUB_BRANCH=<branch> via bazel, so a value like
    # "modular/v26.2" causes nightly content to be excluded.
    nightly_content = ""
    if os.environ.get("GITHUB_BRANCH", "main") == "main":
        with open(nightly_path) as f:
            nightly_lines = f.readlines()
        nightly_content = (
            "".join(_remove_empty_headings(nightly_lines)).rstrip() + "\n"
        )
        # Write the nightly notes to a separate nightly.md file
        with open(os.path.join(changelog_dir, "nightly.md"), "w") as f:
            f.write(nightly_content)

    candidates = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(changelog_dir, "v*.md"))
    ]
    min_version = (0, 24, 1)
    version_files = []
    for fname in candidates:
        ver = _parse_version(fname)
        if ver is None:
            print(
                f"WARNING: skipping '{fname}' in changelog/ "
                f"(filename does not match vX.Y.Z.md pattern)"
            )
        elif ver < min_version:
            continue
        else:
            version_files.append(fname)
    version_files.sort(key=lambda f: _parse_version(f) or (), reverse=True)

    # Assemble: frontmatter/intro, nightly (if present), released versions
    # newest-first, then archive.
    assembled = index_content.rstrip() + "\n\n"
    assembled += _frontmatter_to_heading(nightly_content, edit_title=False)
    for vfile in version_files:
        with open(os.path.join(changelog_dir, vfile)) as f:
            content = _frontmatter_to_heading(f.read())
            assembled += "\n" + content.rstrip() + "\n"

    archive_path = os.path.join(changelog_dir, "archive.md")
    if os.path.exists(archive_path):
        with open(archive_path) as f:
            assembled += "\n" + f.read().rstrip() + "\n"

    with open(index_path, "w") as f:
        f.write(assembled)

    # Each source file uses H1/H2; demote so they nest under the page title.
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


def rewrite_mojo_path_prefix(docs_path: str) -> None:
    """Rewrite '/mojo/' to '/docs/' in hyperlinks and JSX props across all doc files."""
    for root, _, files in os.walk(docs_path):
        for filename in files:
            if not filename.endswith((".md", ".mdx")):
                continue
            file_path = os.path.join(root, filename)
            with open(file_path, "r+") as f:
                content = f.read()
                updated = re.sub(
                    r"(\]\(|href=[\"']|url=[\"'])/mojo/", r"\1/docs/", content
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
    assemble_changelog(sys.argv[1])
    rewrite_mojo_path_prefix(sys.argv[1])
