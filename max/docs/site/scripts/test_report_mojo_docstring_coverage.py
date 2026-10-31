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
"""Unit tests for report-mojo-docstring-coverage.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pytest

_SCRIPT = Path(__file__).resolve().parent / "report-mojo-docstring-coverage.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "report_mojo_docstring_coverage", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cov() -> ModuleType:
    return _load()


def _arg(
    name: str, description: str = "", convention: str = "imm"
) -> dict[str, object]:
    return {"name": name, "description": description, "convention": convention}


def _param(
    name: str, description: str = "", kind: str = "pos"
) -> dict[str, object]:
    return {"name": name, "description": description, "passingKind": kind}


def _fn(
    name: str = "f",
    summary: str = "Does a thing.",
    *,
    args: list[dict[str, object]] | None = None,
    parameters: list[dict[str, object]] | None = None,
    returns: dict[str, object] | None = None,
    raises: bool = False,
    raises_doc: str = "",
    signature: str = "def f()",
) -> dict[str, object]:
    return {
        "kind": "function",
        "name": name,
        "summary": summary,
        "args": args or [],
        "parameters": parameters or [],
        "returns": returns,
        "raises": raises,
        "raisesDoc": raises_doc,
        "signature": signature,
    }


class _Gap(Protocol):
    kind: str
    name: str | None


class _Api(Protocol):
    gaps: Sequence[_Gap]


def _kinds(api: _Api) -> list[tuple[str, str | None]]:
    return [(gap.kind, gap.name) for gap in api.gaps]


def test_complete_function_has_no_gaps(cov: ModuleType) -> None:
    fn = _fn(
        args=[_arg("self"), _arg("x", "The input.")],
        parameters=[_param("T", "The type.")],
        returns={"type": "Int", "doc": "The result."},
        raises=True,
        raises_doc="If it fails.",
    )
    api = cov.check_function(fn, "lib.mod.f")
    assert api.status == "complete"


def test_missing_docstring_reports_only_docstring(cov: ModuleType) -> None:
    fn = _fn(summary="", args=[_arg("x")], returns={"type": "Int", "doc": ""})
    api = cov.check_function(fn, "lib.mod.f")
    assert api.status == "missing"
    assert _kinds(api) == [("docstring", None)]


def test_partial_function_lists_each_gap(cov: ModuleType) -> None:
    fn = _fn(
        args=[_arg("x"), _arg("y", "Documented.")],
        parameters=[_param("T")],
        returns={"type": "Int", "doc": ""},
        raises=True,
    )
    api = cov.check_function(fn, "lib.mod.f")
    assert api.status == "partial"
    assert _kinds(api) == [
        ("arg", "x"),
        ("param", "T"),
        ("returns", None),
        ("raises", None),
    ]


def test_matches_compiler_exemptions(cov: ModuleType) -> None:
    fn = _fn(
        name="__init__",
        args=[_arg("self"), _arg("result", convention="out")],
        parameters=[_param("origin", kind="inferred")],
        returns={"type": "Self", "doc": ""},
    )
    assert cov.check_function(fn, "lib.mod.S.__init__").status == "complete"
    never = _fn(returns={"type": "Never", "doc": ""})
    assert cov.check_function(never, "lib.mod.abort").status == "complete"


def test_gap_ids_distinguish_overloads(cov: ModuleType) -> None:
    first = cov.check_function(
        _fn(args=[_arg("x")], signature="def f(x: Int)"), "lib.mod.f"
    )
    second = cov.check_function(
        _fn(args=[_arg("x")], signature="def f(x: Float64)"), "lib.mod.f"
    )
    assert first.gap_ids() != second.gap_ids()
    assert first.gap_ids()[0].endswith("|arg:x")


def test_walk_package_skips_private_names(cov: ModuleType) -> None:
    package = {
        "kind": "package",
        "name": "lib",
        "modules": [
            {
                "kind": "module",
                "name": "mod",
                "summary": "A module.",
                "aliases": [],
                "functions": [
                    {"overloads": [_fn("public", summary="")]},
                    {"overloads": [_fn("_private", summary="")]},
                ],
                "structs": [
                    {
                        "kind": "struct",
                        "name": "S",
                        "summary": "A struct.",
                        "parameters": [_param("T")],
                        "fields": [{"name": "value", "summary": ""}],
                        "aliases": [{"name": "A", "summary": "An alias."}],
                        "functions": [],
                    }
                ],
                "traits": [
                    {
                        "kind": "trait",
                        "name": "Tr",
                        "summary": "A trait.",
                        "fields": [],
                        "aliases": [{"name": "Element", "summary": ""}],
                        "functions": [],
                    }
                ],
            },
            {"kind": "module", "name": "_impl", "summary": ""},
        ],
        "packages": [{"kind": "package", "name": "_hidden", "modules": []}],
    }
    modules = cov.walk_package(package, "lib")
    assert [m.path for m in modules] == ["lib.mod"]
    gaps = {api.path: _kinds(api) for api in modules[0].apis if api.gaps}
    assert gaps == {
        "lib.mod.public": [("docstring", None)],
        "lib.mod.S": [("param", "T")],
        "lib.mod.S.value": [("docstring", None)],
    }


def test_percent_never_rounds_up_to_100(cov: ModuleType) -> None:
    assert cov._percent(99.99) == "99.9%"
    assert cov._percent(100.0) == "100%"
    assert cov._percent(200 / 3) == "66.6%"


def test_report_end_to_end(
    cov: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bazel passes workspace-relative paths and runs from the runfiles root.
    monkeypatch.chdir(tmp_path)
    package_dir = tmp_path / "max/kernels/src/nn"
    package_dir.mkdir(parents=True)
    (package_dir / "BUILD.bazel").write_text(
        "mojo_library(\n    validate_missing_docs = False,\n)\n"
    )
    json_path = package_dir / "nn.docs.mojodoc.json"
    json_path.write_text(
        json.dumps(
            {
                "decl": {
                    "kind": "package",
                    "name": "nn",
                    "modules": [
                        {
                            "kind": "module",
                            "name": "conv",
                            "summary": "Convolutions.",
                            "functions": [
                                {"overloads": [_fn("good")]},
                                {"overloads": [_fn("bad", summary="")]},
                            ],
                        }
                    ],
                    "packages": [],
                }
            }
        )
    )
    report = cov.build_report(
        [json_path.relative_to(tmp_path)], repo_root=tmp_path
    )
    library = report.libraries[0]
    assert (library.name, library.package) == ("nn", "max/kernels/src/nn")
    assert not library.validation_enabled
    assert (len(report.apis), cov.count(report.apis, "complete")) == (3, 2)

    markdown = cov.render_markdown(report, details=True)
    assert "| `nn`    | off        | 3    | 2        | 0       | 1" in markdown
    assert "- `nn.conv.bad` (function): no docstring" in markdown

    payload = json.loads(cov.render_json(report))
    module = payload["libraries"][0]["modules"][0]
    assert module["incomplete_apis"][0]["path"] == "nn.conv.bad"
