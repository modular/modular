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
"""Reports how many Mojo docs Python and Mojo snippets have tested sources.

Scans Markdown pages under ``Mojo/docs/site/``, extracts Python and Mojo fenced
code blocks, and matches each block against tested files in
``Mojo/docs/site/code/`` and ``# DOC:``-tagged files in ``Mojo/examples/``.

Writes a Markdown report to ``code-coverage.md`` next to this script (or
``--markdown PATH``) and, optionally, JSON (``--json``). Designed to run
with ``python3`` from the repo root so a later GitHub or Buildkite job can
call it without a Bazel build.

Usage:
    python3 Mojo/docs/site/scripts/report-code-coverage.py
    ./bazelw run //Mojo/docs/site/scripts:report-code-coverage

The default run writes ``Mojo/docs/site/scripts/code-coverage.md``. Pass
``--markdown PATH`` to write elsewhere, ``--json PATH`` for JSON, or
``--fail-under PERCENT`` to exit non-zero when coverage is below that value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKDOWN_PATH = SCRIPT_DIR / "code-coverage.md"
REPO_ROOT = Path(
    os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY",
        SCRIPT_DIR.parents[3],
    )
)

SITE_REL = Path("Mojo/docs/site")
CODE_REL = Path("Mojo/docs/site/code")
EXAMPLES_REL = Path("Mojo/examples")
SITE_PREFIX = "Mojo/docs/site/"

# Directories under Mojo/docs/site/ that are not published pages, plus the
# repo README and nightly changelog that sit beside the site content.
PAGE_EXCLUDE_DIRS = frozenset({"code", "releases", "scripts"})
PAGE_EXCLUDE_NAMES = frozenset({"README.md", "nightly-changelog.md"})

CODE_EXTENSIONS = frozenset({".py", ".mojo"})
BLOCK_LANGUAGES = frozenset({"python", "py", "mojo"})

SCHEMA_VERSION = 1
PARTIAL_THRESHOLD = 0.5
# Fences with this many normalized lines or fewer are not required to
# have a test. They are usually fragments, not runnable examples.
MIN_TESTABLE_LINES = 4


# A block does not require a test when any of these lines appear in it
# (after the same whitespace normalization used for matching). Add a row
# when a class of snippet cannot be tested in docs CI. ``py`` is treated
# as ``python``. An empty ``languages`` set applies to every language.
@dataclass(frozen=True)
class SkipTestLine:
    """A line that exempts a docs code block from needing a test."""

    line: str
    languages: frozenset[str]
    reason: str


SKIP_TEST_LINES: tuple[SkipTestLine, ...] = ()

DOC_HEADER_RE = re.compile(r"#\s*DOC:\s*(\S+)")
FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

Category = Literal["no_examples", "no_tests", "with_tests", "skipped_only"]
BlockStatus = Literal["covered", "partial", "uncovered", "skipped"]


@dataclass(frozen=True)
class CodeBlock:
    """A Python or Mojo fence extracted from a docs page."""

    line: int
    language: str
    text: str
    info: str = ""


@dataclass(frozen=True)
class SourceFile:
    """A tested source file that may cover one or more docs blocks."""

    path: Path
    doc_ref: str | None
    lines: tuple[str, ...]
    explicit_header: bool = False


@dataclass
class BlockResult:
    """Match result for one code block on a page."""

    line: int
    language: str
    status: BlockStatus
    match_ratio: float
    source: str | None
    header_warning: str | None = None
    skip_reason: str | None = None


@dataclass
class PageResult:
    """Coverage classification for one docs page."""

    path: str
    category: Category
    blocks: list[BlockResult] = field(default_factory=list)

    @property
    def countable(self) -> list[BlockResult]:
        return [block for block in self.blocks if block.status != "skipped"]

    @property
    def total(self) -> int:
        return len(self.countable)

    @property
    def covered(self) -> int:
        return sum(1 for block in self.countable if block.status == "covered")

    @property
    def partial(self) -> int:
        return sum(1 for block in self.countable if block.status == "partial")

    @property
    def skipped(self) -> int:
        return sum(1 for block in self.blocks if block.status == "skipped")

    @property
    def coverage_percent(self) -> float:
        if not self.countable:
            return 100.0
        return 100.0 * self.covered / self.total

    @property
    def has_partial_only(self) -> bool:
        return self.covered == 0 and self.partial > 0


@dataclass(frozen=True)
class Warning:
    """A source-to-page inconsistency the report should surface."""

    kind: str
    source: str
    doc: str | None = None
    detail: str | None = None


@dataclass
class Report:
    """Full coverage report for the Mojo docs tree."""

    generated_at: datetime
    pages: list[PageResult]
    warnings: list[Warning]

    @property
    def pages_no_examples(self) -> list[PageResult]:
        return [page for page in self.pages if page.category == "no_examples"]

    @property
    def pages_no_tests(self) -> list[PageResult]:
        return [page for page in self.pages if page.category == "no_tests"]

    @property
    def pages_with_tests(self) -> list[PageResult]:
        return [page for page in self.pages if page.category == "with_tests"]

    @property
    def pages_full_coverage(self) -> list[PageResult]:
        return [
            page
            for page in self.pages_with_tests
            if page.covered == page.total and page.total > 0
        ]

    @property
    def pages_some_tests(self) -> list[PageResult]:
        return [
            page for page in self.pages_with_tests if page.covered < page.total
        ]

    @property
    def block_total(self) -> int:
        return sum(page.total for page in self.pages)

    @property
    def block_covered(self) -> int:
        return sum(page.covered for page in self.pages)

    @property
    def block_partial(self) -> int:
        return sum(page.partial for page in self.pages)

    @property
    def block_skipped(self) -> int:
        return sum(page.skipped for page in self.pages)

    @property
    def coverage_percent(self) -> float:
        if self.block_total == 0:
            return 100.0
        return 100.0 * self.block_covered / self.block_total


def discover_pages(site_root: Path) -> list[Path]:
    """Returns Markdown pages under ``site_root``, applying the exclusions."""
    pages: list[Path] = []
    for path in site_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".mdx"}:
            continue
        rel = path.relative_to(site_root)
        if rel.parts and rel.parts[0] in PAGE_EXCLUDE_DIRS:
            continue
        if rel.name in PAGE_EXCLUDE_NAMES:
            continue
        pages.append(path)
    return sorted(pages)


def extract_blocks(text: str) -> list[CodeBlock]:
    """Extracts Python and Mojo fenced blocks, including indented fences."""
    blocks: list[CodeBlock] = []
    for match in FENCE_RE.finditer(text):
        info = match.group("info").strip()
        language = info.split()[0].casefold() if info else ""
        if language not in BLOCK_LANGUAGES:
            continue
        line = text.count("\n", 0, match.start()) + 1
        body = match.group("body")
        indent = match.group("indent")
        if indent:
            stripped = [
                (body_line.removeprefix(indent))
                for body_line in body.splitlines()
            ]
            joined = "\n".join(stripped)
            if body.endswith("\n"):
                joined += "\n"
            body = textwrap.dedent(joined)
        blocks.append(
            CodeBlock(line=line, language=language, text=body, info=info)
        )
    return blocks


def parse_doc_header(text: str) -> str | None:
    """Returns the first ``# DOC:`` page ref, or None if there isn't one.

    Only refs that start with ``Mojo/docs/site/`` belong to this tree; that
    prefix is stripped so callers always see a path relative to the Mojo
    site root. Refs for the MAX docs tree are ignored.
    """
    for line in text.splitlines():
        match = DOC_HEADER_RE.search(line)
        if match is None:
            continue
        ref = match.group(1)
        if ref.startswith(SITE_PREFIX):
            return ref[len(SITE_PREFIX) :]
        return None
    return None


def implicit_doc_ref(
    source_rel: Path, site_root: Path, code_rel: Path = CODE_REL
) -> str | None:
    """Returns the page a ``code/`` file maps to, or None.

    ``code/manual/basics/hello.mojo`` maps to ``manual/basics.mdx`` when
    that page exists. Matches the directory-plus-``.mdx`` rule in
    ``checkExampleDocSync.yaml``, and also accepts ``.md``.
    """
    try:
        under_code = source_rel.relative_to(code_rel)
    except ValueError:
        return None
    parent = under_code.parent
    if not parent.parts:
        return None
    stem = parent.as_posix()
    for suffix in (".mdx", ".md"):
        candidate = f"{stem}{suffix}"
        if (site_root / candidate).is_file():
            return candidate
    return None


def normalize_lines(text: str) -> list[str]:
    """Dedents ``text`` and drops blanks, comments, and extra whitespace."""
    lines: list[str] = []
    for raw in textwrap.dedent(text).splitlines():
        stripped = raw.rstrip()
        if not stripped:
            continue
        if stripped.lstrip().startswith("#"):
            continue
        lines.append(" ".join(stripped.split()))
    return lines


def _canonical_language(language: str) -> str:
    if language == "py":
        return "python"
    return language


def _fence_skip_reason(info: str) -> str | None:
    tokens = info.split()
    for token in tokens[1:]:
        if token == "no-test":
            return "no-test"
        if token.startswith("duplicate-of="):
            return "duplicate-of"
    return None


def skip_reason_for(block: CodeBlock) -> str | None:
    """Returns the skip reason when ``block`` does not require a test."""
    fence_reason = _fence_skip_reason(block.info)
    if fence_reason is not None:
        return fence_reason
    block_lines = normalize_lines(block.text)
    if len(block_lines) < MIN_TESTABLE_LINES:
        return "3 or fewer lines"
    language = _canonical_language(block.language)
    line_set = set(block_lines)
    for rule in SKIP_TEST_LINES:
        if rule.languages and language not in rule.languages:
            continue
        normalized = normalize_lines(rule.line)
        if normalized and normalized[0] in line_set:
            return rule.reason
    return None


def subsequence_hit_count(
    needle: Sequence[str], haystack: Sequence[str]
) -> int:
    """Returns how many ``needle`` lines appear in order in ``haystack``.

    Block lines with no source match are skipped so later lines can still
    count toward coverage.
    """
    if not needle or not haystack:
        return 0
    haystack_index = 0
    hits = 0
    for needle_line in needle:
        search = haystack_index
        while search < len(haystack):
            if haystack[search] == needle_line:
                hits += 1
                haystack_index = search + 1
                break
            search += 1
    return hits


def _source_from_path(
    path: Path, repo_root: Path, site_root: Path, code_rel: Path
) -> SourceFile | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    rel = path.relative_to(repo_root)
    explicit = parse_doc_header(text)
    doc_ref: str | None
    if explicit is not None:
        doc_ref = explicit
        explicit_header = True
    else:
        doc_ref = implicit_doc_ref(rel, site_root, code_rel)
        explicit_header = False
    return SourceFile(
        path=rel,
        doc_ref=doc_ref,
        lines=tuple(normalize_lines(text)),
        explicit_header=explicit_header,
    )


def load_sources(
    repo_root: Path,
    code_root: Path | None = None,
    examples_root: Path | None = None,
    site_root: Path | None = None,
) -> list[SourceFile]:
    """Loads tested sources from ``code/`` and DOC-tagged example files."""
    code_dir = code_root if code_root is not None else repo_root / CODE_REL
    examples_dir = (
        examples_root if examples_root is not None else repo_root / EXAMPLES_REL
    )
    site = site_root if site_root is not None else repo_root / SITE_REL
    sources: list[SourceFile] = []

    if code_dir.is_dir():
        try:
            code_rel = code_dir.relative_to(repo_root)
        except ValueError:
            code_rel = CODE_REL
        for path in sorted(code_dir.rglob("*")):
            if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
                continue
            source = _source_from_path(path, repo_root, site, code_rel)
            if source is not None:
                sources.append(source)

    if examples_dir.is_dir():
        for path in sorted(examples_dir.rglob("*")):
            if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
                continue
            source = _source_from_path(path, repo_root, site, CODE_REL)
            if source is not None and source.doc_ref is not None:
                sources.append(source)

    return sources


def _best_match(
    block_lines: Sequence[str], sources: Sequence[SourceFile]
) -> tuple[SourceFile, float] | None:
    best: tuple[SourceFile, float] | None = None
    for source in sources:
        hits = subsequence_hit_count(block_lines, source.lines)
        ratio = hits / len(block_lines) if block_lines else 0.0
        if best is None or ratio > best[1]:
            best = (source, ratio)
    return best


def match_block(
    block: CodeBlock, page_ref: str, sources: Sequence[SourceFile]
) -> BlockResult:
    """Matches ``block`` against ``sources``, preferring this page's files."""
    reason = skip_reason_for(block)
    if reason is not None:
        return BlockResult(
            line=block.line,
            language=block.language,
            status="skipped",
            match_ratio=0.0,
            source=None,
            skip_reason=reason,
        )
    block_lines = normalize_lines(block.text)
    preferred = [src for src in sources if src.doc_ref == page_ref]
    others = [src for src in sources if src.doc_ref != page_ref]

    best_preferred = _best_match(block_lines, preferred)
    best_other = _best_match(block_lines, others)

    chosen: tuple[SourceFile, float] | None
    header_warning: str | None = None
    if best_preferred and (
        best_other is None or best_preferred[1] >= best_other[1]
    ):
        chosen = best_preferred
    else:
        chosen = best_other
        if best_other is not None and best_other[1] > 0:
            other, _ratio = best_other
            if other.explicit_header and other.doc_ref != page_ref:
                header_warning = "wrong_doc_header"

    if chosen is None or chosen[1] == 0:
        return BlockResult(
            line=block.line,
            language=block.language,
            status="uncovered",
            match_ratio=0.0,
            source=None,
        )

    source, ratio = chosen
    if ratio >= 1.0:
        status: BlockStatus = "covered"
    elif ratio >= PARTIAL_THRESHOLD:
        status = "partial"
    else:
        status = "uncovered"
        header_warning = None

    return BlockResult(
        line=block.line,
        language=block.language,
        status=status,
        match_ratio=ratio,
        source=str(source.path) if status != "uncovered" else None,
        header_warning=header_warning if status != "uncovered" else None,
    )


def classify_page(page_ref: str, blocks: Sequence[BlockResult]) -> PageResult:
    """Assigns a page to no-examples, no-tests, with-tests, or skipped-only."""
    countable = [block for block in blocks if block.status != "skipped"]
    if not blocks:
        category: Category = "no_examples"
    elif not countable:
        category = "skipped_only"
    elif any(block.status in {"covered", "partial"} for block in countable):
        category = "with_tests"
    else:
        category = "no_tests"
    return PageResult(path=page_ref, category=category, blocks=list(blocks))


def collect_warnings(
    pages: Sequence[PageResult],
    sources: Sequence[SourceFile],
    page_refs: Iterable[str],
) -> list[Warning]:
    """Finds stale DOC headers, missing pages, and header mismatches."""
    known_pages = set(page_refs)
    used_by_page: dict[str, set[str]] = {ref: set() for ref in known_pages}
    warnings: list[Warning] = []

    for page in pages:
        for block in page.blocks:
            if block.source:
                used_by_page.setdefault(page.path, set()).add(block.source)
            if block.header_warning == "wrong_doc_header" and block.source:
                warnings.append(
                    Warning(
                        kind="wrong_doc_header",
                        source=block.source,
                        doc=page.path,
                        detail=(
                            "matched a block on this page but its # DOC: "
                            "header points elsewhere"
                        ),
                    )
                )

    seen_stale: set[str] = set()
    for source in sources:
        if not source.explicit_header or source.doc_ref is None:
            continue
        rel = str(source.path)
        if source.doc_ref not in known_pages:
            warnings.append(
                Warning(
                    kind="missing_page",
                    source=rel,
                    doc=source.doc_ref,
                    detail="# DOC: header names a page that was not found",
                )
            )
            continue
        if (
            rel not in used_by_page.get(source.doc_ref, set())
            and rel not in seen_stale
        ):
            seen_stale.add(rel)
            warnings.append(
                Warning(
                    kind="stale_source",
                    source=rel,
                    doc=source.doc_ref,
                    detail="matches no Python or Mojo block on its page",
                )
            )

    return warnings


def build_report(
    repo_root: Path,
    site_root: Path | None = None,
    code_root: Path | None = None,
    examples_root: Path | None = None,
) -> Report:
    """Builds the coverage report for the tree under ``repo_root``."""
    site = site_root if site_root is not None else repo_root / SITE_REL
    page_paths = discover_pages(site)
    sources = load_sources(repo_root, code_root, examples_root, site)

    pages: list[PageResult] = []
    page_refs: list[str] = []
    for path in page_paths:
        page_ref = path.relative_to(site).as_posix()
        page_refs.append(page_ref)
        blocks = extract_blocks(path.read_text(encoding="utf-8"))
        results = [match_block(block, page_ref, sources) for block in blocks]
        pages.append(classify_page(page_ref, results))

    return Report(
        generated_at=datetime.now(timezone.utc),
        pages=pages,
        warnings=collect_warnings(pages, sources, page_refs),
    )


def _percent(value: float) -> str:
    if value == int(value):
        return f"{int(value)}%"
    return f"{value:.1f}%"


def _append_page_list(
    lines: list[str], pages: Sequence[PageResult], *, show_counts: bool
) -> None:
    if not pages:
        lines.append("None.")
        lines.append("")
        return
    for page in pages:
        if show_counts:
            extra = " (includes partial tests)" if page.has_partial_only else ""
            lines.append(
                f"- `{page.path}` ({page.covered}/{page.total}){extra}"
            )
        else:
            lines.append(f"- `{page.path}`")
    lines.append("")


def render_markdown(report: Report) -> str:
    """Renders ``report`` as a grouped Markdown document."""
    lines = [
        "# Docs code coverage",
        "",
        f"Generated at {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "## Summary",
        "",
        (
            f"- {report.block_covered}/{report.block_total} "
            f"({_percent(report.coverage_percent)}) code blocks tested"
        ),
        f"- {len(report.pages_no_tests)} pages with no tests",
        f"- {report.block_skipped} skipped blocks",
        "",
        "## Pages with no tests",
        "",
    ]
    _append_page_list(lines, report.pages_no_tests, show_counts=False)

    lines.extend(["## Pages with some tests", ""])
    _append_page_list(lines, report.pages_some_tests, show_counts=True)

    lines.extend(["## Pages with full coverage", ""])
    _append_page_list(lines, report.pages_full_coverage, show_counts=True)

    tested = report.pages_with_tests
    if tested:
        lines.extend(["## Block details", ""])
        for page in tested:
            extra = " (includes partial tests)" if page.has_partial_only else ""
            lines.append(
                f"### `{page.path}` ({page.covered}/{page.total}){extra}"
            )
            lines.append("")
            for block in page.blocks:
                lines.append(f"- {_block_line(block)}")
            lines.append("")

    untested = report.pages_no_tests
    if untested:
        lines.extend(["## Untested blocks", ""])
        for page in untested:
            lines.append(f"### `{page.path}` (0/{page.total})")
            lines.append("")
            for block in page.blocks:
                lines.append(f"- {_block_line(block)}")
            lines.append("")

    lines.extend(["## Warnings", ""])
    if not report.warnings:
        lines.append("None.")
        lines.append("")
    else:
        for warning in report.warnings:
            doc = f" (`{warning.doc}`)" if warning.doc else ""
            detail = f": {warning.detail}" if warning.detail else ""
            lines.append(f"- `{warning.source}`{doc}{detail}")
        lines.append("")

    return "\n".join(lines)


def _block_line(block: BlockResult) -> str:
    if block.status == "skipped":
        reason = block.skip_reason or "skip-test line"
        return f"L{block.line} {block.language}: skipped ({reason})"
    ratio = _percent(100.0 * block.match_ratio)
    source = f" — `{block.source}`" if block.source else ""
    return f"L{block.line} {block.language}: {block.status} ({ratio}){source}"


def render_json(report: Report) -> str:
    """Renders ``report`` as a versioned JSON document."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": report.generated_at.isoformat(),
        "summary": {
            "pages": len(report.pages),
            "pages_no_examples": len(report.pages_no_examples),
            "pages_no_tests": len(report.pages_no_tests),
            "pages_with_tests": len(report.pages_with_tests),
            "blocks": report.block_total,
            "blocks_covered": report.block_covered,
            "blocks_partial": report.block_partial,
            "blocks_skipped": report.block_skipped,
            "blocks_uncovered": (
                report.block_total - report.block_covered - report.block_partial
            ),
            "coverage_percent": report.coverage_percent,
        },
        "pages": [
            {
                "path": page.path,
                "category": page.category,
                "covered": page.covered,
                "partial": page.partial,
                "total": page.total,
                "coverage_percent": page.coverage_percent,
                "blocks": [
                    {
                        "line": block.line,
                        "language": block.language,
                        "status": block.status,
                        "match_ratio": block.match_ratio,
                        "source": block.source,
                        "skip_reason": block.skip_reason,
                    }
                    for block in page.blocks
                ],
            }
            for page in report.pages
        ],
        "warnings": [
            {
                "kind": warning.kind,
                "source": warning.source,
                "doc": warning.doc,
                "detail": warning.detail,
            }
            for warning in report.warnings
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN_PATH,
        help=(
            "Write the Markdown report to this path (default: "
            "code-coverage.md next to this script)."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write a JSON report to this path. Omitted means no JSON.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="PERCENT",
        help=(
            "Exit 1 when overall covered/total block coverage is below "
            "this percent. Off by default."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: inferred from this file's location).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report = build_report(repo_root)
    markdown = render_markdown(report)

    # Regenerated code-coverage.md must pass MD013 (80-column prose); run
    # ./bazelw run //:format before committing updates to that file.
    args.markdown.write_text(markdown, encoding="utf-8")

    if args.json is not None:
        args.json.write_text(render_json(report), encoding="utf-8")

    below_floor = (
        args.fail_under is not None
        and report.coverage_percent < args.fail_under
    )
    if below_floor:
        print(
            f"Coverage {report.coverage_percent:.1f}% is below "
            f"--fail-under {args.fail_under}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
