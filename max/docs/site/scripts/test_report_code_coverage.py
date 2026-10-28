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
"""Unit tests for report-code-coverage.py."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent / "report-code-coverage.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "report_code_coverage", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cov() -> ModuleType:
    return _load()


def test_extract_blocks_filters_language(cov: ModuleType) -> None:
    text = """\
Intro.

```python
print("py")
```

```mojo
print("mojo")
```

```bash
echo hi
```

```text
not code
```
"""
    blocks = cov.extract_blocks(text)
    assert [(b.language, b.line) for b in blocks] == [
        ("python", 3),
        ("mojo", 7),
    ]


def test_extract_blocks_keeps_fence_attributes(cov: ModuleType) -> None:
    text = """\
```python title="foo.py" {11-17}
x = 1
```
"""
    blocks = cov.extract_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert blocks[0].line == 1
    assert blocks[0].text == "x = 1\n"


def test_extract_blocks_handles_indented_fences(cov: ModuleType) -> None:
    text = """\
1. First step:

   ```python
   value = 2
   ```
"""
    blocks = cov.extract_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert "value = 2" in blocks[0].text
    assert not blocks[0].text.startswith("   ")


def test_normalize_lines_drops_comments_and_blanks(cov: ModuleType) -> None:
    text = """\
# license
# DOC: develop/broadcasting.mdx

from max.tensor import Tensor

# Output:
#   [Dim(4)]

activations = Tensor.zeros([4, 8])
"""
    assert cov.normalize_lines(text) == [
        "from max.tensor import Tensor",
        "activations = Tensor.zeros([4, 8])",
    ]


def test_match_block_covers_excerpt_without_comments(cov: ModuleType) -> None:
    source = cov.SourceFile(
        path=Path("max/docs/site/code/develop/broadcasting/auto_broadcast.py"),
        doc_ref="develop/broadcasting.mdx",
        lines=tuple(
            cov.normalize_lines(
                """\
# license header
# DOC: develop/broadcasting.mdx

# Output:
#   [Dim(4), Dim(8)]

from max.tensor import Tensor

activations = Tensor.zeros([4, 8])
bias = Tensor.ones([8])
result = activations + bias
print(result.shape)
"""
            )
        ),
    )
    block = cov.CodeBlock(
        line=86,
        language="python",
        text="""\
from max.tensor import Tensor

activations = Tensor.zeros([4, 8])
bias = Tensor.ones([8])
result = activations + bias
print(result.shape)
""",
    )
    result = cov.match_block(block, "develop/broadcasting.mdx", [source])
    assert result.status == "covered"
    assert result.match_ratio == 1.0
    assert result.source == str(source.path)


def test_match_block_partial_when_half_the_lines_hit(cov: ModuleType) -> None:
    source = cov.SourceFile(
        path=Path("max/docs/site/code/develop/example.py"),
        doc_ref="develop/example.mdx",
        lines=("from max.tensor import Tensor", "print(x.shape)"),
    )
    block = cov.CodeBlock(
        line=10,
        language="python",
        text="""\
from max.tensor import Tensor
x = Tensor.zeros([2])
y = x + x
print(x.shape)
""",
    )
    result = cov.match_block(block, "develop/example.mdx", [source])
    assert result.status == "partial"
    assert result.match_ratio == 0.5


def test_match_block_prefers_doc_tagged_source(cov: ModuleType) -> None:
    tagged = cov.SourceFile(
        path=Path("max/docs/site/code/develop/page/a.py"),
        doc_ref="develop/page.mdx",
        lines=("alpha = 1", "beta = 2", "gamma = 3", "delta = 4"),
    )
    other = cov.SourceFile(
        path=Path("max/examples/other.py"),
        doc_ref="develop/other.mdx",
        lines=("alpha = 1", "beta = 2", "gamma = 3", "delta = 4"),
    )
    block = cov.CodeBlock(
        line=4,
        language="python",
        text="alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\n",
    )
    result = cov.match_block(block, "develop/page.mdx", [other, tagged])
    assert result.status == "covered"
    assert result.source == str(tagged.path)
    assert result.header_warning is None


def test_parse_doc_header_strips_site_prefix(cov: ModuleType) -> None:
    assert (
        cov.parse_doc_header("# DOC: develop/basic-ops.mdx extra")
        == "develop/basic-ops.mdx"
    )
    assert (
        cov.parse_doc_header("# DOC: max/docs/site/develop/basic-ops.mdx")
        == "develop/basic-ops.mdx"
    )
    assert cov.parse_doc_header("# DOC: Mojo/docs/site/manual.mdx") is None
    assert cov.parse_doc_header("print(1)") is None


def test_discover_pages_excludes_configured_trees(
    cov: ModuleType, tmp_path: Path
) -> None:
    site = tmp_path / "site"
    (site / "develop").mkdir(parents=True)
    (site / "code" / "develop").mkdir(parents=True)
    (site / "_includes").mkdir()
    (site / "releases").mkdir()
    (site / "scripts").mkdir()
    (site / "develop" / "page.mdx").write_text("# Page\n")
    (site / "README.md").write_text("# readme\n")
    (site / "README.INTERNAL.md").write_text("# internal\n")
    (site / "code" / "develop" / "notes.md").write_text("# notes\n")
    (site / "_includes" / "install.mdx").write_text("# include\n")
    (site / "releases" / "v26.1.md").write_text("# release\n")
    (site / "scripts" / "README.md").write_text("# scripts\n")

    pages = [p.relative_to(site).as_posix() for p in cov.discover_pages(site)]
    assert pages == ["develop/page.mdx"]


def test_classify_page_categories(cov: ModuleType) -> None:
    empty = cov.classify_page("empty.mdx", [])
    assert empty.category == "no_examples"

    uncovered = cov.BlockResult(
        line=1,
        language="python",
        status="uncovered",
        match_ratio=0.0,
        source=None,
    )
    assert cov.classify_page("none.mdx", [uncovered]).category == "no_tests"

    partial = cov.BlockResult(
        line=1,
        language="python",
        status="partial",
        match_ratio=0.5,
        source="max/docs/site/code/a.py",
    )
    assert cov.classify_page("some.mdx", [partial]).category == "with_tests"

    skipped = cov.BlockResult(
        line=1,
        language="python",
        status="skipped",
        match_ratio=0.0,
        source=None,
        skip_reason="live inference client",
    )
    only_skipped = cov.classify_page("openai.mdx", [skipped])
    assert only_skipped.category == "skipped_only"
    assert only_skipped.total == 0
    mixed = cov.classify_page("mixed.mdx", [skipped, uncovered])
    assert mixed.category == "no_tests"
    assert mixed.total == 1


def test_openai_import_skips_inference_block(cov: ModuleType) -> None:
    block = cov.CodeBlock(
        line=117,
        language="python",
        text="""\
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="EMPTY",
)
""",
    )
    result = cov.match_block(block, "container.mdx", [])
    assert result.status == "skipped"
    assert result.skip_reason == "live inference client"

    py_alias = cov.CodeBlock(
        line=10,
        language="py",
        text=(
            "from openai import OpenAI\n"
            "client = OpenAI(api_key='EMPTY')\n"
            "print(client)\n"
            "print(1)\n"
        ),
    )
    assert cov.match_block(py_alias, "serve/text-to-text.mdx", []).status == (
        "skipped"
    )


def test_short_block_is_skipped(cov: ModuleType) -> None:
    block = cov.CodeBlock(
        line=833,
        language="mojo",
        text="# INCORRECT\nshared_counter[0] += my_value\n",
    )
    result = cov.match_block(block, "gpu/block-and-warp.mdx", [])
    assert result.status == "skipped"
    assert result.skip_reason == "3 or fewer lines"


def test_markdown_defaults_to_script_dir(
    cov: ModuleType, tmp_path: Path
) -> None:
    assert cov.DEFAULT_MARKDOWN_PATH == _SCRIPT.parent / "code-coverage.md"

    site = tmp_path / "max" / "docs" / "site"
    (site / "develop").mkdir(parents=True)
    (site / "develop" / "empty.mdx").write_text("# Empty\n")
    (site / "code").mkdir(parents=True)
    (tmp_path / "max" / "examples").mkdir(parents=True)

    default_path = tmp_path / "code-coverage.md"
    exit_code = cov.main(
        [
            "--repo-root",
            str(tmp_path),
            "--markdown",
            str(default_path),
        ]
    )
    assert exit_code == 0
    assert default_path.is_file()
    assert default_path.read_text().startswith("# Docs code coverage\n")


def test_json_is_optional(cov: ModuleType, tmp_path: Path) -> None:
    site = tmp_path / "max" / "docs" / "site"
    (site / "develop").mkdir(parents=True)
    (site / "develop" / "empty.mdx").write_text("# Empty\n")
    (tmp_path / "max" / "docs" / "site" / "code").mkdir(parents=True)
    (tmp_path / "max" / "examples").mkdir(parents=True)

    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    exit_code = cov.main(
        [
            "--repo-root",
            str(tmp_path),
            "--markdown",
            str(markdown_path),
        ]
    )
    assert exit_code == 0
    assert markdown_path.is_file()
    assert not json_path.exists()

    exit_code = cov.main(
        [
            "--repo-root",
            str(tmp_path),
            "--markdown",
            str(markdown_path),
            "--json",
            str(json_path),
        ]
    )
    assert exit_code == 0
    assert json_path.is_file()
    payload = json_path.read_text()
    assert '"schema_version": 1' in payload


def test_render_markdown_lists_pages_before_summary(cov: ModuleType) -> None:
    uncovered = cov.BlockResult(
        line=1,
        language="python",
        status="uncovered",
        match_ratio=0.0,
        source=None,
    )
    covered = cov.BlockResult(
        line=2,
        language="python",
        status="covered",
        match_ratio=1.0,
        source="max/docs/site/code/a.py",
    )
    report = cov.Report(
        generated_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
        pages=[
            cov.classify_page("empty.mdx", []),
            cov.classify_page("none.mdx", [uncovered]),
            cov.classify_page("partial.mdx", [covered, uncovered]),
            cov.classify_page("full.mdx", [covered]),
        ],
        warnings=[],
    )
    text = cov.render_markdown(report)
    assert text.index("## Summary") < text.index("## Pages with no tests")
    assert text.index("- 2/4 (50%) code blocks tested") < text.index(
        "## Pages with no tests"
    )
    assert text.index("- 1 pages with no tests") < text.index(
        "## Pages with no tests"
    )
    assert text.index("## Pages with no tests") < text.index(
        "## Pages with some tests"
    )
    assert text.index("## Pages with some tests") < text.index(
        "## Pages with full coverage"
    )
    assert text.index("`none.mdx`") < text.index("`partial.mdx`")
    assert text.index("`partial.mdx` (1/2)") < text.index("`full.mdx` (1/1)")
    assert text.count("## Summary") == 1
    assert "## No code examples" not in text

    zero_covered = cov.BlockResult(
        line=1,
        language="mojo",
        status="partial",
        match_ratio=0.92,
        source="max/examples/gpu-block-and-warp/tiled_matmul.mojo",
    )
    zero_page = cov.classify_page("gpu/block-and-warp.mdx", [zero_covered])
    labeled = cov.render_markdown(
        cov.Report(
            generated_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
            pages=[zero_page],
            warnings=[],
        )
    )
    assert "`gpu/block-and-warp.mdx` (0/1) (includes partial tests)" in labeled
