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
"""Generate the supported models MDX table from architecture source files.

Parses each architecture's arch.py to extract SupportedArchitecture metadata,
then generates a Markdown table and writes it into models.mdx.

Usage:
    python generate-models-table.py
"""

import argparse
import ast
import os
import sys
from collections.abc import Set
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

# Under `bazel run`, __file__ resolves into the runfiles tree instead of the
# real workspace.  BUILD_WORKSPACE_DIRECTORY is set by Bazel and points to the
# actual repo root, so prefer it when available.
REPO_ROOT = Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", SCRIPT_DIR.parent.parent.parent)
)

ARCH_BASE = REPO_ROOT / "max" / "python" / "max" / "pipelines" / "architectures"

INIT_FILE = ARCH_BASE / "__init__.py"

OUTPUT_FILE = REPO_ROOT / "oss" / "modular" / "docs" / "max" / "models.mdx"

GITHUB_ARCH_URL = (
    "https://github.com/modular/modular/tree/main/"
    "max/python/max/pipelines/architectures"
)

OUTPUT_LABELS: dict[str, str] = {
    "TEXT_GENERATION": "text",
    "EMBEDDINGS_GENERATION": "embeddings",
    "AUDIO_GENERATION": "audio",
    "SPEECH_TOKEN_GENERATION": "speech-tokens",
    "PIXEL_GENERATION": "image",
}


def derive_modality_labels(
    task: str | None, input_modalities: Set[str]
) -> list[str]:
    """Derive human-readable input-to-output modality labels.

    Uses the ``task`` enum name for the output type and the explicit
    ``input_modalities`` set (declared on each ``SupportedArchitecture``)
    for the input types.
    """
    output = OUTPUT_LABELS.get(task, task) if task else "unknown"
    labels: list[str] = []
    if "TEXT" in input_modalities:
        labels.append(f"text-to-{output}")
    if "IMAGE" in input_modalities:
        labels.append(f"image-to-{output}")
    if "VIDEO" in input_modalities:
        labels.append(f"video-to-{output}")
    if not labels:
        raise ValueError(
            f"Architecture has no recognized input modalities for task"
            f" {task!r}. Check the input_modalities declaration in arch.py."
        )
    return labels


def parse_init_imports() -> list[tuple[str, list[str]]]:
    """Parse __init__.py to discover (module_name, [variable_names]) tuples."""
    if not INIT_FILE.exists():
        raise FileNotFoundError(
            f"Architecture __init__.py not found at {INIT_FILE}."
        )
    tree = ast.parse(INIT_FILE.read_text())

    # Find the register_all_models function
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "register_all_models"
        ):
            func_body = node.body
            break
    else:
        raise RuntimeError(
            "Could not find register_all_models() in __init__.py"
        )

    # Collect all `from .X import Y` statements
    imports: list[tuple[str, list[str]]] = []
    for stmt in func_body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            names = [alias.name for alias in stmt.names]
            imports.append((stmt.module, names))

    # Also find which variable names appear in the `architectures = [...]` list
    registered_names: set[str] = set()
    for stmt in func_body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "architectures"
                ):
                    if isinstance(stmt.value, ast.List):
                        for elt in stmt.value.elts:
                            if isinstance(elt, ast.Name):
                                registered_names.add(elt.id)

    # Filter imports to only include variables that are in the architectures list
    filtered: list[tuple[str, list[str]]] = []
    for module, names in imports:
        registered = [n for n in names if n in registered_names]
        if registered:
            filtered.append((module, registered))

    return filtered


def extract_keyword_value(
    call_node: ast.Call, keyword_name: str
) -> ast.expr | None:
    """Extract a keyword argument value from an ast.Call node."""
    for kw in call_node.keywords:
        if kw.arg == keyword_name:
            return kw.value
    return None


def _resolve_module_level_collections(
    tree: ast.Module,
) -> dict[str, list[str]]:
    """Build a map of variable_name -> list[str] for module-level list/set assignments."""
    result: dict[str, list[str]] = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, (ast.List, ast.Set)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                ]
                result[target.id] = values
    return result


def parse_arch_file(arch_path: Path) -> list[dict[str, Any]]:
    """Parse an arch.py file and extract all SupportedArchitecture definitions."""
    tree = ast.parse(arch_path.read_text())
    module_collections = _resolve_module_level_collections(tree)
    results = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func = call.func

        # Match SupportedArchitecture(...)
        is_supported_arch = False
        if isinstance(func, ast.Name) and func.id == "SupportedArchitecture":
            is_supported_arch = True
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "SupportedArchitecture"
        ):
            is_supported_arch = True

        if not is_supported_arch:
            continue

        # Get variable name
        var_name = None
        if node.targets and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id

        # Extract fields
        name_node = extract_keyword_value(call, "name")
        name = name_node.value if isinstance(name_node, ast.Constant) else None

        example_node = extract_keyword_value(call, "example_repo_ids")
        example_repo_ids: list[str] = []
        if isinstance(example_node, (ast.List, ast.Set)):
            for elt in example_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    example_repo_ids.append(elt.value)
        elif (
            isinstance(example_node, ast.Name)
            and example_node.id in module_collections
        ):
            example_repo_ids = list(module_collections[example_node.id])

        modality_node = extract_keyword_value(call, "task")
        modality = None
        if isinstance(modality_node, ast.Attribute):
            modality = modality_node.attr

        enc_node = extract_keyword_value(call, "supported_encodings")
        supported_encodings: set[str] = set()
        if isinstance(enc_node, (ast.Set, ast.List)):
            for elt in enc_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    supported_encodings.add(elt.value)
        elif (
            isinstance(enc_node, ast.Name) and enc_node.id in module_collections
        ):
            supported_encodings = set(module_collections[enc_node.id])
        supported_encodings = {
            e for e in supported_encodings if not e.startswith("q4_")
        }

        modalities_node = extract_keyword_value(call, "input_modalities")
        input_modalities: set[str] = {"TEXT"}
        if isinstance(modalities_node, (ast.Set, ast.List)):
            parsed = set()
            for elt in modalities_node.elts:
                if isinstance(elt, ast.Attribute):
                    parsed.add(elt.attr)
            if parsed:
                input_modalities = parsed

        multi_gpu_node = extract_keyword_value(call, "multi_gpu_supported")
        multi_gpu = False
        if isinstance(multi_gpu_node, ast.Constant):
            multi_gpu = bool(multi_gpu_node.value)

        if name is None:
            continue

        results.append(
            {
                "var_name": var_name,
                "name": name,
                "example_repo_ids": example_repo_ids,
                "modality": modality,
                "input_modalities": input_modalities,
                "supported_encodings": supported_encodings,
                "multi_gpu_supported": multi_gpu,
            }
        )

    return results


def collect_architectures() -> list[dict[str, Any]]:
    """Collect all registered architecture metadata."""
    imports = parse_init_imports()
    all_archs: list[dict[str, Any]] = []

    for module_name, var_names in imports:
        arch_path = ARCH_BASE / module_name.replace(".", "/") / "arch.py"
        if not arch_path.exists():
            print(
                f"Warning: {arch_path} does not exist, skipping",
                file=sys.stderr,
            )
            continue

        parsed = parse_arch_file(arch_path)
        for arch in parsed:
            if arch["var_name"] in var_names:
                arch["module_name"] = module_name
                all_archs.append(arch)

    return all_archs


def filter_architectures(archs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply filtering rules to remove duplicates and helper architectures."""
    # Build set of all architecture names for cross-referencing
    all_names = {a["name"] for a in archs}

    filtered = []
    for arch in archs:
        name = arch["name"]

        # Exclude speculative decoding helpers (Eagle/NextN)
        if "Eagle" in name or "NextN" in name:
            continue

        # Exclude ModuleV3 duplicates where a base counterpart exists
        if name.endswith("_ModuleV3"):
            base_name = name[: -len("_ModuleV3")]
            if base_name in all_names:
                continue
            # No base counterpart (e.g., Flux models) — keep it

        filtered.append(arch)

    return filtered


def merge_architectures(archs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge entries that share the same display name into single rows.

    Combines example repos, modality/context pairs, encodings, and multi-GPU
    support so that architectures registered under the same HuggingFace model
    name (e.g. ``Qwen3ForCausalLM`` for both text-gen and embeddings) produce
    one table row.
    """
    grouped: dict[str, dict[str, Any]] = {}
    insert_order: list[str] = []

    for arch in archs:
        display_name = arch["name"].removesuffix("_ModuleV3")

        if display_name not in grouped:
            insert_order.append(display_name)
            grouped[display_name] = {
                "name": display_name,
                "module_name": arch["module_name"],
                "example_repo_ids": list(arch["example_repo_ids"]),
                "modality_input_pairs": {
                    (arch["modality"], frozenset(arch["input_modalities"]))
                },
                "supported_encodings": set(arch["supported_encodings"]),
                "multi_gpu_supported": arch["multi_gpu_supported"],
            }
        else:
            entry = grouped[display_name]
            entry["example_repo_ids"].extend(arch["example_repo_ids"])
            entry["modality_input_pairs"].add(
                (arch["modality"], frozenset(arch["input_modalities"]))
            )
            entry["supported_encodings"] |= arch["supported_encodings"]
            entry["multi_gpu_supported"] = (
                entry["multi_gpu_supported"] or arch["multi_gpu_supported"]
            )

    return [grouped[name] for name in insert_order]


def format_table(archs: list[dict[str, Any]]) -> str:
    """Generate an HTML table from architecture metadata."""
    archs = sorted(archs, key=lambda a: a["name"].lower())
    ind = "  "

    lines = [
        '<table className="models-table">',
        f"{ind}<thead>",
        f"{ind}{ind}<tr>",
        f"{ind}{ind}{ind}<th>Architecture</th>",
        f"{ind}{ind}{ind}<th>Example models (repo IDs)</th>",
        f"{ind}{ind}{ind}<th>Modality</th>",
        f"{ind}{ind}{ind}<th>Encodings</th>",
        f"{ind}{ind}{ind}<th>Multi-GPU</th>",
        f"{ind}{ind}</tr>",
        f"{ind}</thead>",
        f"{ind}<tbody>",
    ]

    for arch in archs:
        display_name = arch["name"]

        # Format example repos as HuggingFace links (deduplicated, order-preserving)
        repo_links = []
        seen_repos: set[str] = set()
        for repo_id in arch["example_repo_ids"]:
            if repo_id in seen_repos:
                continue
            seen_repos.add(repo_id)
            repo_links.append(
                f'{ind}{ind}{ind}{ind}<a href="https://huggingface.co/{repo_id}">{repo_id}</a>'
            )
        examples_inner = ",<br/>\n".join(repo_links)

        # Derive modality labels from all (task, input_modalities) pairs
        all_labels: set[str] = set()
        for task, input_mods in sorted(
            arch["modality_input_pairs"],
            key=lambda p: (p[0] or ""),
        ):
            all_labels.update(derive_modality_labels(task, input_mods))
        modality_cell = (
            ",<br/>".join(sorted(all_labels)) if all_labels else "Unknown"
        )

        encodings = ", ".join(sorted(arch["supported_encodings"]))

        multi_gpu = "Yes" if arch["multi_gpu_supported"] else "No"

        arch_url = f"{GITHUB_ARCH_URL}/{arch['module_name']}"

        lines.append(f"{ind}{ind}<tr>")
        lines.append(
            f"{ind}{ind}{ind}<td className='arch'>"
            f'<a href="{arch_url}"><code>{display_name}</code></a></td>'
        )
        lines.append(f"{ind}{ind}{ind}<td className='models'>")
        lines.append(examples_inner)
        lines.append(f"{ind}{ind}{ind}</td>")
        lines.append(
            f"{ind}{ind}{ind}<td className='modality'>{modality_cell}</td>"
        )
        lines.append(
            f"{ind}{ind}{ind}<td className='encodings'>{encodings}</td>"
        )
        lines.append(f"{ind}{ind}{ind}<td className='gpus'>{multi_gpu}</td>")
        lines.append(f"{ind}{ind}</tr>")

    lines.append(f"{ind}</tbody>")
    lines.append("</table>")

    return "\n".join(lines)


BEGIN_MARKER = "{/* BEGIN TABLE */}"
END_MARKER = "{/* END TABLE */}"


def write_table(table: str) -> None:
    """Replace content between BEGIN/END TABLE markers in models.mdx."""
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Output file not found at {OUTPUT_FILE}. "
            "Is the script running from the repository root?"
        )
    existing = OUTPUT_FILE.read_text()

    begin_idx = existing.find(BEGIN_MARKER)
    if begin_idx == -1:
        raise RuntimeError(
            f"Marker {BEGIN_MARKER!r} not found in {OUTPUT_FILE}"
        )

    end_idx = existing.find(END_MARKER)
    if end_idx == -1:
        raise RuntimeError(
            f"Marker {END_MARKER!r} not found in {OUTPUT_FILE}. "
            "Add it after the table to protect trailing content."
        )

    before = existing[: begin_idx + len(BEGIN_MARKER)]
    after = existing[end_idx:]
    content = before + "\n\n" + table + "\n\n" + after
    OUTPUT_FILE.write_text(content)
    print(f"Wrote {OUTPUT_FILE}")


def check_table(table: str) -> bool:
    """Return True if models.mdx already contains the expected table."""
    if not OUTPUT_FILE.exists():
        return False
    existing = OUTPUT_FILE.read_text()

    begin_idx = existing.find(BEGIN_MARKER)
    if begin_idx == -1:
        return False

    end_idx = existing.find(END_MARKER)
    if end_idx == -1:
        return False

    current = existing[begin_idx + len(BEGIN_MARKER) : end_idx]
    expected = "\n\n" + table + "\n\n"
    return current == expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the supported models table for models.mdx"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that models.mdx is up-to-date; exit 1 if not.",
    )
    args = parser.parse_args()

    all_archs = collect_architectures()
    filtered = filter_architectures(all_archs)
    merged = merge_architectures(filtered)
    table = format_table(merged)

    already_current = check_table(table)

    if args.check:
        if already_current:
            print("✅ models.mdx is up-to-date.")
            sys.exit(0)
        else:
            print(
                "❌ models.mdx is out-of-date.\n"
                "Run `./bazelw run //oss/modular/docs:generate-models-table` "
                "to regenerate, then commit the updated models.mdx file.",
                file=sys.stderr,
            )
            sys.exit(1)

    if already_current:
        print("✅ models.mdx is already up-to-date.")
    else:
        write_table(table)
        print("✅ models.mdx updated.")


if __name__ == "__main__":
    main()
