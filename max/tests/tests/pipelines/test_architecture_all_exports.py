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
"""Validates that each architecture package re-exports its public classes.

Each architecture package under ``max.pipelines.architectures`` should list
every public class defined in ``arch.py``, ``model.py``, and ``model_config.py``
in its ``__init__.py`` ``__all__``. The re-exports drive the generated API
reference pages: members not in ``__all__`` are silently omitted from the
rendered docs.

This test parses files with :mod:`ast` (no imports) so it runs without
pulling in the full pipelines dependency graph.
"""

from __future__ import annotations

import ast
from pathlib import Path

import max.pipelines.architectures as architectures_pkg
import pytest

_ARCH_ROOT = Path(architectures_pkg.__file__).parent
_CHECKED_SUBMODULES = ("arch.py", "model.py", "model_config.py")


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None


def _public_classes(path: Path) -> list[str]:
    tree = _parse(path)
    if tree is None:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def _arch_vars(path: Path) -> list[str]:
    tree = _parse(path)
    if tree is None:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_arch"):
                    names.append(target.id)
    return names


def _get_all(init_path: Path) -> list[str] | None:
    tree = _parse(init_path)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                return [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                ]
    return None


def _is_architecture_package(pkg_dir: Path) -> bool:
    """A registered architecture has ``arch.py`` with a SupportedArchitecture()."""
    arch = pkg_dir / "arch.py"
    if not arch.exists():
        return False
    return "SupportedArchitecture(" in arch.read_text()


def _is_shim_package(pkg_dir: Path) -> bool:
    """A shim package has neither ``model.py`` nor ``model_config.py``.

    These packages only register a :class:`SupportedArchitecture` on top of
    another architecture (for example, ``exaone`` extends ``llama3``) and
    don't define their own model or config classes, so there's nothing to
    require in ``__all__``.
    """
    return (
        not (pkg_dir / "model.py").exists()
        and not (pkg_dir / "model_config.py").exists()
    )


def _architecture_packages() -> list[Path]:
    return sorted(
        p
        for p in _ARCH_ROOT.iterdir()
        if p.is_dir()
        and (p / "__init__.py").exists()
        and _is_architecture_package(p)
        and not _is_shim_package(p)
    )


@pytest.mark.parametrize(
    "pkg_dir", _architecture_packages(), ids=lambda p: p.name
)
def test_architecture_exports_public_classes(pkg_dir: Path) -> None:
    """Every public class in arch/model/model_config must be in ``__all__``."""
    init_all = _get_all(pkg_dir / "__init__.py")
    assert init_all is not None, (
        f"{pkg_dir.name}/__init__.py must define __all__"
    )
    exported = set(init_all)

    expected: dict[str, str] = {}
    for submodule in _CHECKED_SUBMODULES:
        submodule_path = pkg_dir / submodule
        for cls in _public_classes(submodule_path):
            expected.setdefault(cls, submodule)
    for var in _arch_vars(pkg_dir / "arch.py"):
        expected.setdefault(var, "arch.py")

    missing = sorted(n for n in expected if n not in exported)
    assert not missing, (
        f"{pkg_dir.name}/__init__.py: __all__ is missing "
        f"{', '.join(f'{n} (from {expected[n]})' for n in missing)}. "
        "Add these names to __all__ so they render on the generated API "
        "reference page."
    )
