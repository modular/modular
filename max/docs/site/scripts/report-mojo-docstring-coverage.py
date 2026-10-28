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
"""Reports how completely published Mojo APIs are documented.

Reads the ``mojo doc`` JSON for each published Mojo library (the MAX kernel
libraries and the ``max`` package) and checks every public API for a
docstring plus a description of each argument, parameter, return value, and
raised error. The rules match ``mojo doc --diagnose-missing-doc-strings``, so
a library with no gaps here passes once its ``validate_missing_docs`` is
turned on.

Writes a Markdown report to ``mojo-docstring-coverage.md`` next to this
script (or ``--markdown PATH``) and, optionally, JSON with every gap
(``--json``).

Usage:
    ./bazelw run //max/docs/site/scripts:report-mojo-docstring-coverage

Pass ``--details`` to list every incomplete API in the Markdown report.
Positional arguments are ``mojo doc`` JSON files; the Bazel target supplies
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCRIPT_REL = Path("max/docs/site/scripts")
REPO_ROOT = Path(
    os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY",
        Path(__file__).resolve().parent.parent.parent.parent.parent,
    )
)
DEFAULT_MARKDOWN_NAME = "mojo-docstring-coverage.md"
MOJODOC_SUFFIX = ".docs.mojodoc.json"

SCHEMA_VERSION = 1

VALIDATION_OFF_RE = re.compile(r"validate_missing_docs\s*=\s*False")

# Return types that do not require a ``Returns:`` section.
NO_RESULT_TYPES = frozenset({"", "None", "Never"})

GapKind = Literal["docstring", "arg", "param", "returns", "raises"]
ApiStatus = Literal["complete", "partial", "missing"]


@dataclass(frozen=True)
class Gap:
    """One missing piece of documentation on an API."""

    kind: GapKind
    name: str | None = None

    def label(self) -> str:
        if self.kind == "docstring":
            return "no docstring"
        if self.name is None:
            return f"no {self.kind} description"
        return f"{self.kind} `{self.name}`"


@dataclass
class Api:
    """A public API and the documentation it is missing."""

    path: str
    kind: str
    signature: str = ""
    gaps: list[Gap] = field(default_factory=list)

    @property
    def status(self) -> ApiStatus:
        if not self.gaps:
            return "complete"
        if any(gap.kind == "docstring" for gap in self.gaps):
            return "missing"
        return "partial"

    def gap_ids(self) -> list[str]:
        """Returns stable IDs for this API's gaps.

        The signature hash keeps overloads distinct without depending on
        their order in the source file.
        """
        digest = hashlib.sha1(self.signature.encode("utf-8")).hexdigest()[:8]
        ids = []
        for gap in self.gaps:
            suffix = f":{gap.name}" if gap.name else ""
            ids.append(f"{self.path}|{digest}|{gap.kind}{suffix}")
        return ids


@dataclass
class Module:
    """A public module and its APIs, including the module itself."""

    path: str
    apis: list[Api] = field(default_factory=list)


@dataclass
class Library:
    """A Bazel ``mojo_library`` with published docs."""

    name: str
    package: str
    validation_enabled: bool
    modules: list[Module]

    @property
    def apis(self) -> list[Api]:
        return [api for module in self.modules for api in module.apis]


@dataclass
class Report:
    """Docstring coverage across all published Mojo libraries."""

    generated_at: datetime
    libraries: list[Library]

    @property
    def apis(self) -> list[Api]:
        return [api for library in self.libraries for api in library.apis]


def count(apis: Sequence[Api], status: ApiStatus) -> int:
    return sum(1 for api in apis if api.status == status)


def complete_percent(apis: Sequence[Api]) -> float:
    if not apis:
        return 100.0
    return 100.0 * count(apis, "complete") / len(apis)


JsonObject = Mapping[str, object]


def _objects(decl: JsonObject, key: str) -> list[JsonObject]:
    value = decl.get(key, [])
    assert isinstance(value, list)
    return value


def _str(decl: JsonObject, key: str) -> str:
    value = decl.get(key)
    return value if isinstance(value, str) else ""


def is_private(name: str) -> bool:
    """Returns whether ``name`` is private (``_x`` but not ``__x__``)."""
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def is_private_module(name: str) -> bool:
    return name.startswith("_") and name != "__init__"


def _param_gaps(decl: JsonObject) -> list[Gap]:
    # Inferred parameters (declared before `//`) are optional, as in the
    # compiler's check.
    return [
        Gap("param", _str(param, "name"))
        for param in _objects(decl, "parameters")
        if not param.get("description")
        and param.get("passingKind") != "inferred"
    ]


def check_function(fn: JsonObject, path: str) -> Api:
    """Checks one function overload against the compiler's rules."""
    api = Api(path=path, kind="function", signature=_str(fn, "signature"))
    if not fn.get("summary"):
        api.gaps.append(Gap("docstring"))
        return api
    for arg in _objects(fn, "args"):
        name = _str(arg, "name")
        if name == "self" or arg.get("convention") == "out":
            continue
        if not arg.get("description"):
            api.gaps.append(Gap("arg", name))
    api.gaps.extend(_param_gaps(fn))
    returns = fn.get("returns") or {}
    assert isinstance(returns, Mapping)
    if (
        _str(returns, "type") not in NO_RESULT_TYPES
        and not returns.get("doc")
        and fn.get("name") != "__init__"
    ):
        api.gaps.append(Gap("returns"))
    if fn.get("raises") and not fn.get("raisesDoc"):
        api.gaps.append(Gap("raises"))
    return api


def _check_simple(decl: JsonObject, path: str, kind: str) -> Api:
    api = Api(path=path, kind=kind, signature=_str(decl, "signature"))
    if not decl.get("summary"):
        api.gaps.append(Gap("docstring"))
        return api
    api.gaps.extend(_param_gaps(decl))
    return api


def _public(decls: Iterable[JsonObject]) -> list[JsonObject]:
    return [decl for decl in decls if not is_private(_str(decl, "name"))]


def _functions(decl: JsonObject) -> list[JsonObject]:
    overloads = [
        fn
        for overload_set in _objects(decl, "functions")
        for fn in _objects(overload_set, "overloads")
    ]
    return _public(overloads)


def check_type(decl: JsonObject, path: str) -> list[Api]:
    """Checks a struct or trait and its fields, aliases, and methods."""
    kind = _str(decl, "kind")
    apis = [_check_simple(decl, path, kind)]
    for fld in _public(_objects(decl, "fields")):
        apis.append(_check_simple(fld, f"{path}.{_str(fld, 'name')}", "field"))
    # Trait `comptime` members aren't checked by the compiler.
    if kind == "struct":
        for alias in _public(_objects(decl, "aliases")):
            apis.append(
                _check_simple(alias, f"{path}.{_str(alias, 'name')}", "alias")
            )
    for fn in _functions(decl):
        apis.append(check_function(fn, f"{path}.{_str(fn, 'name')}"))
    return apis


def check_module(decl: JsonObject, path: str) -> Module:
    """Checks a module docstring and every public API it declares."""
    module = Module(path=path)
    module.apis.append(_check_simple(decl, path, "module"))
    for alias in _public(_objects(decl, "aliases")):
        module.apis.append(
            _check_simple(alias, f"{path}.{_str(alias, 'name')}", "alias")
        )
    for fn in _functions(decl):
        module.apis.append(check_function(fn, f"{path}.{_str(fn, 'name')}"))
    types = _objects(decl, "structs") + _objects(decl, "traits")
    for type_decl in _public(types):
        module.apis.extend(
            check_type(type_decl, f"{path}.{_str(type_decl, 'name')}")
        )
    return module


def walk_package(decl: JsonObject, path: str) -> list[Module]:
    """Returns every public module under a package, depth first."""
    modules: list[Module] = []
    for child in _objects(decl, "modules"):
        name = _str(child, "name")
        if not is_private_module(name):
            modules.append(check_module(child, f"{path}.{name}"))
    for child in _objects(decl, "packages"):
        name = _str(child, "name")
        if not name.startswith("_"):
            modules.extend(walk_package(child, f"{path}.{name}"))
    return modules


def load_library(json_path: Path, repo_root: Path) -> Library:
    """Loads one library's ``mojo doc`` JSON.

    ``json_path`` is relative to the workspace root, as Bazel's ``rootpaths``
    gives it (for example ``max/kernels/src/nn/nn.docs.mojodoc.json``), so its
    directory is the library's Bazel package.
    """
    package = json_path.parent.as_posix()
    build_file = (repo_root / package / "BUILD.bazel").read_text(
        encoding="utf-8"
    )
    root = json.loads(json_path.read_text(encoding="utf-8"))["decl"]
    return Library(
        name=json_path.name.removesuffix(MOJODOC_SUFFIX),
        package=package,
        validation_enabled=VALIDATION_OFF_RE.search(build_file) is None,
        modules=walk_package(root, str(root["name"])),
    )


def build_report(json_paths: Sequence[Path], repo_root: Path) -> Report:
    libraries = [load_library(path, repo_root) for path in json_paths]
    libraries.sort(key=lambda library: library.name)
    return Report(generated_at=datetime.now(timezone.utc), libraries=libraries)


def _percent(value: float) -> str:
    # Truncate rather than round so only a complete library shows 100%.
    value = math.floor(value * 10) / 10
    if value == int(value):
        return f"{int(value)}%"
    return f"{value:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Returns a Markdown table aligned the way rumdl's MD060 expects."""
    widths = [
        max(len(cell) for cell in column)
        for column in zip(headers, *rows, strict=True)
    ]

    def line(cells: Sequence[str]) -> str:
        padded = (
            cell.ljust(width) for cell, width in zip(cells, widths, strict=True)
        )
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return [line(headers), separator, *(line(row) for row in rows)]


def _counts(apis: Sequence[Api]) -> list[str]:
    return [
        str(len(apis)),
        str(count(apis, "complete")),
        str(count(apis, "partial")),
        str(count(apis, "missing")),
        _percent(complete_percent(apis)),
    ]


def _summary(apis: Sequence[Api]) -> dict[str, int | float]:
    return {
        "apis": len(apis),
        "complete": count(apis, "complete"),
        "partial": count(apis, "partial"),
        "missing": count(apis, "missing"),
        "complete_percent": complete_percent(apis),
    }


_COUNT_HEADERS = ["APIs", "Complete", "Partial", "Missing", "%"]


def render_markdown(report: Report, *, details: bool = False) -> str:
    """Renders ``report`` as Markdown, optionally listing every gap."""
    apis = report.apis
    complete_libraries = [
        lib for lib in report.libraries if complete_percent(lib.apis) == 100
    ]
    lines = [
        "# Mojo docstring coverage",
        "",
        f"Generated at {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "An API is complete when it has a docstring and describes every",
        "argument, parameter, return value, and raised error that",
        "`mojo doc --diagnose-missing-doc-strings` requires. Partial APIs",
        "have a docstring with some of those missing; missing APIs have no",
        "docstring.",
        "",
        "## Summary",
        "",
        (
            f"- {count(apis, 'complete')}/{len(apis)} "
            f"({_percent(complete_percent(apis))}) public APIs complete"
        ),
        f"- {count(apis, 'partial')} partial",
        f"- {count(apis, 'missing')} with no docstring",
        (
            f"- {len(complete_libraries)}/{len(report.libraries)} "
            "libraries complete"
        ),
        "",
        "## Libraries",
        "",
    ]
    lines.extend(
        _table(
            ["Library", "Validation", *_COUNT_HEADERS],
            [
                [
                    f"`{library.name}`",
                    "on" if library.validation_enabled else "off",
                    *_counts(library.apis),
                ]
                for library in report.libraries
            ],
        )
    )
    lines.append("")

    for library in report.libraries:
        lines.extend([f"## `{library.name}`", ""])
        lines.append(f"Bazel package: `//{library.package}`.")
        lines.append("")
        incomplete = sorted(
            (m for m in library.modules if complete_percent(m.apis) < 100),
            key=lambda m: (complete_percent(m.apis), m.path),
        )
        done = len(library.modules) - len(incomplete)
        lines.append(f"{done}/{len(library.modules)} modules complete.")
        lines.append("")
        if not incomplete:
            continue
        lines.extend(
            _table(
                ["Module", *_COUNT_HEADERS],
                [[f"`{m.path}`", *_counts(m.apis)] for m in incomplete],
            )
        )
        lines.append("")

        if not details:
            continue
        for module in incomplete:
            lines.extend([f"### `{module.path}`", ""])
            for api in module.apis:
                if api.gaps:
                    gaps = ", ".join(gap.label() for gap in api.gaps)
                    lines.append(f"- `{api.path}` ({api.kind}): {gaps}")
            lines.append("")

    return "\n".join(lines)


def render_json(report: Report) -> str:
    """Renders ``report`` as versioned JSON that lists every gap."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": report.generated_at.isoformat(),
        "summary": _summary(report.apis),
        "libraries": [
            {
                "name": library.name,
                "package": library.package,
                "validation_enabled": library.validation_enabled,
                **_summary(library.apis),
                "modules": [
                    {
                        "path": module.path,
                        **_summary(module.apis),
                        "incomplete_apis": [
                            {
                                "path": api.path,
                                "kind": api.kind,
                                "status": api.status,
                                "gaps": [gap.label() for gap in api.gaps],
                                "gap_ids": api.gap_ids(),
                            }
                            for api in module.apis
                            if api.gaps
                        ],
                    }
                    for module in library.modules
                ],
            }
            for library in report.libraries
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mojodoc_json",
        nargs="+",
        type=Path,
        help="`mojo doc` JSON files, one per library.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help=(
            f"Write the Markdown report to this path (default: "
            f"{DEFAULT_MARKDOWN_NAME} next to this script)."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write a JSON report with every gap to this path.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="List every incomplete API in the Markdown report.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root, used to read each library's BUILD.bazel.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report = build_report(args.mojodoc_json, repo_root)

    # Relative output paths are relative to where the user ran the command,
    # not the runfiles directory `bazel run` executes in.
    cwd = Path(os.environ.get("BUILD_WORKING_DIRECTORY", Path.cwd()))
    markdown_path = (
        args.markdown or repo_root / SCRIPT_REL / DEFAULT_MARKDOWN_NAME
    )
    (cwd / markdown_path).write_text(
        render_markdown(report, details=args.details), encoding="utf-8"
    )
    if args.json is not None:
        (cwd / args.json).write_text(render_json(report), encoding="utf-8")

    apis = report.apis
    print(
        f"{count(apis, 'complete')}/{len(apis)} "
        f"({_percent(complete_percent(apis))}) public APIs complete.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
