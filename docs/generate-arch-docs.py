#!/usr/bin/env python3
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
"""Generate per-architecture API doc RST files and keep navigation in sync.

Scans the pipeline architecture registry, extracts metadata from each
arch.py, and generates:
  - One RST page per registered architecture
  - The architectures index RST (pipelines.architectures.rst)
  - Toctree entries in index.rst
  - Sidebar items under max.pipelines.architectures in sidebars.json

Usage:
    ./bazelw run //oss/modular/docs:generate-arch-docs          # regenerate
    ./bazelw run //oss/modular/docs:generate-arch-docs -- --check  # CI validation
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", SCRIPT_DIR.parent.parent.parent)
)
ARCH_BASE = REPO_ROOT / "max" / "python" / "max" / "pipelines" / "architectures"
DOCS_DIR = REPO_ROOT / "max" / "python" / "docs"
INDEX_RST = DOCS_DIR / "index.rst"
SIDEBARS_JSON = REPO_ROOT / "oss" / "modular" / "docs" / "sidebars.json"

# Maps PipelineTask enum values to human-readable category headings.
# Ordered by display priority on the index page.
TASK_CATEGORIES: dict[str, str] = {
    "TEXT_GENERATION": "Text generation",
    "EMBEDDINGS_GENERATION": "Embeddings",
    "PIXEL_GENERATION": "Image generation",
    "AUDIO_GENERATION": "Audio generation",
    "SPEECH_TOKEN_GENERATION": "Speech token generation",
}


def _detect_task(dir_name: str) -> str:
    """Return the PipelineTask enum member name from an architecture's arch.py."""
    arch_file = ARCH_BASE / dir_name / "arch.py"
    if not arch_file.exists():
        return "TEXT_GENERATION"

    tree = ast.parse(arch_file.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "task"
            and isinstance(node.value, ast.Attribute)
        ):
            return node.value.attr
    return "TEXT_GENERATION"


def discover_categories(dir_names: list[str]) -> dict[str, list[str]]:
    """Group architectures by PipelineTask, returning {category: [dirs]}."""
    groups: dict[str, list[str]] = {}
    for d in dir_names:
        task = _detect_task(d)
        category = TASK_CATEGORIES.get(task, task)
        groups.setdefault(category, []).append(d)
    # Return in TASK_CATEGORIES display order, then any unknown categories.
    ordered: dict[str, list[str]] = {}
    for category in TASK_CATEGORIES.values():
        if category in groups:
            ordered[category] = sorted(groups.pop(category))
    for category in sorted(groups):
        ordered[category] = sorted(groups[category])
    return ordered


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------


def discover_registered_dirs() -> list[str]:
    """Return sorted directory names of all registered architectures."""
    tree = ast.parse((ARCH_BASE / "__init__.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "register_all_models"
        ):
            modules = set()
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module:
                    modules.add(stmt.module)
            return sorted(modules)
    raise RuntimeError("Could not find register_all_models() in __init__.py")


# ---------------------------------------------------------------------------
# Content generation
# ---------------------------------------------------------------------------


def generate_rst(dir_name: str) -> str:
    """Generate RST content for one architecture page."""
    mod = f"max.pipelines.architectures.{dir_name}"

    lines = [
        f":title: {mod}",
        f":sidebar_label: {dir_name}",
        ":type: module",
        ":lang: python",
        ":wrapper_class: rst-module-autosummary",
        "",
        mod,
        "=" * len(mod),
        "",
        f".. automodule:: {mod}",
        "   :members:",
        "   :imported-members:",
        "   :show-inheritance:",
    ]

    lines.append("")
    return "\n".join(lines)


def generate_index_rst(
    dir_names: list[str], categories: dict[str, list[str]]
) -> str:
    """Generate the architectures index page."""
    lines = [
        ":title: max.pipelines.architectures",
        ":type: module",
        ":lang: python",
        ":wrapper_class: rst-module-autosummary",
        "",
        "max.pipelines.architectures",
        "============================",
        "",
        ".. automodule:: max.pipelines.architectures",
        "   :no-members:",
        "",
        "MAX includes built-in support for a wide range of model architectures. Each",
        "architecture module registers a",
        ":class:`~max.pipelines.lib.registry.SupportedArchitecture` instance that tells",
        "the pipeline system how to load, configure, and execute a particular model",
        "family.",
        "",
        # Hidden toctree for sidebar navigation
        ".. toctree::",
        "   :hidden:",
        "",
    ]
    for d in sorted(dir_names):
        lines.append(f"   pipelines.architectures.{d}")

    # Autosummary tables per category
    for category, members in categories.items():
        if not members:
            continue
        lines += [
            "",
            category,
            "-" * len(category),
            "",
            ".. autosummary::",
            "   :nosignatures:",
            "",
        ]
        for d in members:
            lines.append(f"   ~max.pipelines.architectures.{d}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File sync helpers (check or write, controlled by --check flag)
# ---------------------------------------------------------------------------


def sync_file(path: Path, expected: str, check: bool, stale: list[str]) -> None:
    """Compare *path* to *expected*. In check mode, record staleness. Otherwise write."""
    if path.exists() and path.read_text() == expected:
        return
    if check:
        stale.append(path.name)
    else:
        path.write_text(expected)
        print(f"  wrote {path.name}")


def _replace_between(
    text: str, before: str, after: str, new_content: str
) -> str:
    """Replace text between *before* marker (inclusive end) and *after* marker (exclusive start)."""
    bi = text.find(before)
    ai = text.find(after)
    if bi == -1 or ai == -1:
        raise RuntimeError(f"Markers not found: {before!r} / {after!r}")
    return text[: bi + len(before)] + new_content + text[ai:]


def sync_index_rst(dir_names: list[str], check: bool, stale: list[str]) -> None:
    """Keep architecture entries in index.rst in sync."""
    entries = "".join(
        f"   pipelines.architectures.{d}\n" for d in sorted(dir_names)
    )
    text = INDEX_RST.read_text()
    expected = _replace_between(
        text, "   pipelines.architectures\n", "   pipelines.core", entries
    )
    if text == expected:
        return
    if check:
        stale.append("index.rst")
    else:
        INDEX_RST.write_text(expected)
        print("  updated index.rst")


def sync_sidebars_json(
    dir_names: list[str], check: bool, stale: list[str]
) -> None:
    """Keep architecture doc IDs in sidebars.json (Python reference sidebar).

    Expects a category object with label \"max.pipelines.architectures\" under
    the \"Python\" category in maxReferenceSidebar. Inserts that category
    before \"max.profiler\" if it is missing.
    """
    path = SIDEBARS_JSON
    original_text = path.read_text(encoding="utf-8")
    data: object = json.loads(original_text)
    if not isinstance(data, dict):
        raise RuntimeError("sidebars.json: root value must be a JSON object")

    max_ref = data.get("maxReferenceSidebar")
    if not isinstance(max_ref, list):
        raise RuntimeError(
            "sidebars.json: maxReferenceSidebar missing or not a list"
        )

    python_cat: dict[str, object] | None = None
    for entry in max_ref:
        if (
            isinstance(entry, dict)
            and entry.get("label") == "Python"
            and entry.get("type") == "category"
        ):
            python_cat = entry
            break
    if python_cat is None:
        raise RuntimeError(
            'sidebars.json: category with label "Python" not found'
        )

    py_items = python_cat.get("items")
    if not isinstance(py_items, list):
        raise RuntimeError('sidebars.json: Python category "items" not a list')

    new_doc_ids = [
        f"max/api/python/pipelines.architectures.{d}" for d in sorted(dir_names)
    ]

    arch_cat: dict[str, object] | None = None
    for entry in py_items:
        if (
            isinstance(entry, dict)
            and entry.get("label") == "max.pipelines.architectures"
        ):
            arch_cat = entry
            break

    if arch_cat is None:
        insert_at = len(py_items)
        for i, entry in enumerate(py_items):
            if isinstance(entry, dict) and entry.get("label") == "max.profiler":
                insert_at = i
                break
        arch_cat = {
            "type": "category",
            "label": "max.pipelines.architectures",
            "link": {
                "type": "doc",
                "id": "max/api/python/pipelines.architectures",
            },
            "items": new_doc_ids,
        }
        py_items.insert(insert_at, arch_cat)
    elif arch_cat.get("items") == new_doc_ids:
        return
    else:
        arch_cat["items"] = new_doc_ids

    new_text = json.dumps(data, indent=2) + "\n"
    if new_text == original_text:
        return
    if check:
        stale.append("sidebars.json")
    else:
        path.write_text(new_text, encoding="utf-8")
        print("  updated sidebars.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate architecture API docs."
    )
    parser.add_argument("--check", action="store_true", help="Exit 1 if stale.")
    args = parser.parse_args()

    dir_names = discover_registered_dirs()
    categories = discover_categories(dir_names)
    stale: list[str] = []

    # Per-architecture RST pages
    for d in dir_names:
        rst_path = DOCS_DIR / f"pipelines.architectures.{d}.rst"
        sync_file(rst_path, generate_rst(d), args.check, stale)

    # Index RST, index.rst toctree, sidebars.json
    sync_file(
        DOCS_DIR / "pipelines.architectures.rst",
        generate_index_rst(dir_names, categories),
        args.check,
        stale,
    )
    sync_index_rst(dir_names, args.check, stale)
    sync_sidebars_json(dir_names, args.check, stale)

    if args.check:
        if stale:
            print(
                "❌ Architecture docs are out-of-date.\n"
                "Stale files:\n"
                + "\n".join(f"  - {f}" for f in stale)
                + "\n\nRun `./bazelw run //oss/modular/docs:generate-arch-docs`"
                " to regenerate.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("✅ Architecture docs are up-to-date.")
    else:
        print("✅ Architecture docs generated.")


if __name__ == "__main__":
    main()
