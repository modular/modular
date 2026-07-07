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
"""Sybil collection for ``max/python/max`` docstring examples."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sybil import Sybil
from sybil.document import DOCSTRING_PUNCTUATION, PythonDocStringDocument
from sybil.example import Example, NotEvaluated
from sybil.integration.pytest import SybilItem
from sybil.parsers.rest import PythonCodeBlockParser, SkipParser
from sybil.text import LineNumberOffsets

#: Paths under ``max/python/max`` that are never collected.
COLLECT_EXCLUDES = (
    "serve/schemas/",  # generated
    "graph/weights/load_gguf.py",  # hard top-level ``import gguf``
    # benchmark/benchmark_shared/server_metrics.py has a module-level import
    # cycle, and Sybil imports each document's module directly; the benchmark
    # tree's examples are illustrative, so skip the whole directory.
    "benchmark/",
)


def is_private_name(name: str) -> bool:
    """True for a single-underscore name; dunders are public."""
    stem = name.removesuffix(".py")
    return stem.startswith("_") and not (
        stem.startswith("__") and stem.endswith("__")
    )


def _node_char_span(
    offsets: LineNumberOffsets, node: ast.stmt
) -> tuple[int, int] | None:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    start = offsets.get(node.lineno - 1, node.col_offset)
    end = offsets.get(node.end_lineno - 1, node.end_col_offset)
    return start, end


def _private_spans(
    tree: ast.AST, offsets: LineNumberOffsets
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and is_private_name(node.name):
            span = _node_char_span(offsets, node)
            if span is not None:
                spans.append(span)
    return spans


def _is_string_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_inner_span(
    source: str, offsets: LineNumberOffsets, node: ast.Constant
) -> tuple[int, int] | None:
    # Mirror Sybil's own docstring extraction (sybil.document): strip the
    # opening quote and prefix, then an equal-length closing quote. A docstring
    # is always a single string literal, so open and close lengths match.
    # Return None (skip) rather than crash if the literal uses a prefix Sybil's
    # regex doesn't recognize (e.g. ``u`` or ``rf``); such docstrings are rare.
    end_lineno = node.end_lineno or node.lineno
    node_start = offsets.get(node.lineno - 1, node.col_offset)
    node_end = offsets.get(end_lineno - 1, node.end_col_offset or 0)
    punc = DOCSTRING_PUNCTUATION.match(source, node_start, node_end)
    if punc is None:
        return None
    return punc.end(), node_end - len(punc.group(1))


def _extra_docstrings(
    source: str, tree: ast.AST, offsets: LineNumberOffsets
) -> Iterator[tuple[int, int, str]]:
    """Yield ``__doc__`` assignments and attribute docstrings Sybil misses."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "__doc__"
            and _is_string_constant(node.value)
        ):
            assert isinstance(node.value, ast.Constant)
            text = node.value.value
            assert isinstance(text, str)
            span = _string_inner_span(source, offsets, node.value)
            if span is not None:
                yield span[0], span[1], text
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for prev, stmt in zip(node.body, node.body[1:], strict=False):
                if not (
                    isinstance(prev, (ast.Assign, ast.AnnAssign))
                    and isinstance(stmt, ast.Expr)
                    and _is_string_constant(stmt.value)
                ):
                    continue
                assert isinstance(stmt.value, ast.Constant)
                text = stmt.value.value
                assert isinstance(text, str)
                span = _string_inner_span(source, offsets, stmt.value)
                if span is not None:
                    yield span[0], span[1], text


class _PublicDocStringDocument(PythonDocStringDocument):
    @staticmethod
    def extract_docstrings(source: str) -> Iterator[tuple[int, int, str]]:
        tree = ast.parse(source)
        offsets = LineNumberOffsets(source)
        private = _private_spans(tree, offsets)
        docstrings = list(PythonDocStringDocument.extract_docstrings(source))
        docstrings.extend(_extra_docstrings(source, tree, offsets))
        docstrings.sort()
        for start, end, text in docstrings:
            if not any(lo <= start < hi for lo, hi in private):
                yield start, end, text

    def import_document(self, example: Example) -> None:
        # Run examples in isolation. Skip Sybil's default import-region
        # evaluation and do not seed the module's namespace, so each example
        # must import the names it uses. A non-self-contained example then
        # fails here instead of silently borrowing the module's imports, which
        # is what keeps the rendered examples copy-paste runnable.
        self.pop_evaluator(self.import_document)
        raise NotEvaluated()


#: The scanned package root. Sybil anchors ``path`` at the directory of the
#: file that constructs it -- this file's directory, not the scanned tree --
#: and matches collected files by exact path prefix (``Path.relative_to``).
#: Normalize the anchor lexically: bazel runfiles are symlink forests, so
#: ``resolve()`` would escape the sandbox and never match pytest's paths,
#: while a literal ``..`` segment in the anchor never matches anything.
_SCAN_ROOT = Path(os.path.normpath(Path(__file__).parent / "../../python/max"))

_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    path=str(_SCAN_ROOT),
    patterns=["*.py"],
    # Directory excludes need a glob suffix for Sybil's matcher.
    excludes=[
        exclude + "*" if exclude.endswith("/") else exclude
        for exclude in COLLECT_EXCLUDES
    ],
    document_types={".py": _PublicDocStringDocument},
).pytest()


def pytest_collect_file(file_path: Path, parent: Any) -> Any:
    """Collect docstring examples from public modules under max/python/max.

    ``_collect_file`` already restricts to files under ``_SCAN_ROOT`` via
    ``Sybil.should_parse``; this only adds the private-module filter, which
    Sybil's fnmatch excludes can't express (they'd also catch dunder files).
    """
    if is_private_name(file_path.name):
        return None
    return _collect_file(file_path, parent)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail fast if an opted-in target collected zero docstring examples.

    Runs only in sessions that load this plugin via ``-p sybil_collect``.
    """
    examples = [i for i in session.items if isinstance(i, SybilItem)]
    if not examples:
        raise pytest.UsageError(
            "test_docstring_examples is enabled but no docstring examples were "
            "collected; check the target's sources and COLLECT_EXCLUDES."
        )
