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
"""Tests that recurrent speculative architectures disable prefix caching.

The recurrent cache group checkpoints only when the context length lands
exactly on a page boundary, and under the overlap pipeline that length does
not yet include a speculative step's accepted drafts. Prefix caching would
publish checkpoints at the wrong depth, so it must stay off.

Sources are parsed with :mod:`ast` rather than imported.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

_SPEC_MIXIN = "_UnifiedSpecDecodeModelMixin"
"""Base class every unified speculative pipeline model mixes in."""

_STATE_PARAMS = "RecurrentStateParams"
"""Constructed by the architecture that declares a recurrent state cache."""

_FLAG = "enable_prefix_caching"

_KNOWN_RECURRENT_SPEC_ARCHES = frozenset(
    {
        "unified_dflash2_qwen3_5",
        "unified_mtp_inkling",
        "unified_mtp_qwen3_5",
    }
)
"""Architectures that speculate over a recurrent state.

Extend this and the deps in ``BUILD.bazel`` when adding one.
"""


def _arch_root() -> Path:
    """Returns the architectures source directory, in runfiles or the tree."""
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        roots = sorted(
            Path(srcdir).glob("*/max/python/max/pipelines/architectures")
        )
        if roots:
            return roots[0]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = (
            parent / "max" / "python" / "max" / "pipelines" / "architectures"
        )
        if candidate.is_dir():
            return candidate
    raise AssertionError(f"could not locate the architectures tree from {here}")


_ARCH_ROOT = _arch_root()


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None


def _is_speculative(package: Path) -> bool:
    """Returns whether the package's model mixes in the spec-decode driver."""
    tree = _parse(package / "model.py")
    if tree is None:
        return False
    return any(
        isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == _SPEC_MIXIN
            for base in node.bases
        )
        for node in ast.walk(tree)
    )


def _sibling_packages(package: Path) -> set[str]:
    """Returns this package and the sibling packages its model imports."""
    names = {package.name}
    tree = _parse(package / "model.py")
    if tree is None:
        return names
    for node in ast.walk(tree):
        # `from ..qwen3_5.model import ...`
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _declares_recurrent_state(package_name: str) -> bool:
    """Returns whether a package constructs ``RecurrentStateParams``."""
    package = _ARCH_ROOT / package_name
    if not package.is_dir():
        return False
    for path in sorted(package.glob("*.py")):
        tree = _parse(path)
        if tree is None:
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == _STATE_PARAMS
            for node in ast.walk(tree)
        ):
            return True
    return False


def _recurrent_spec_packages() -> list[Path]:
    """Returns the architecture packages that speculate over recurrent state."""
    return [
        package
        for package in sorted(_ARCH_ROOT.iterdir())
        if package.is_dir()
        and _is_speculative(package)
        and any(map(_declares_recurrent_state, _sibling_packages(package)))
    ]


def _prefix_caching_argument(package: Path) -> bool | None:
    """Returns the package's required ``enable_prefix_caching``, or ``None``."""
    tree = _parse(package / "arch.py")
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "required_arguments":
                continue
            if not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(
                keyword.value.keys, keyword.value.values, strict=True
            ):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == _FLAG
                    and isinstance(value, ast.Constant)
                ):
                    assert isinstance(value.value, bool)
                    return value.value
    return None


def test_discovery_finds_the_known_recurrent_spec_architectures() -> None:
    """Checks discovery finds every known recurrent speculative architecture."""
    found = {package.name for package in _recurrent_spec_packages()}

    assert _KNOWN_RECURRENT_SPEC_ARCHES <= found, (
        f"not discovered: {sorted(_KNOWN_RECURRENT_SPEC_ARCHES - found)}."
        " Check this target's deps."
    )


@pytest.mark.parametrize(
    "package", _recurrent_spec_packages(), ids=lambda p: p.name
)
def test_a_recurrent_speculative_architecture_disables_prefix_caching(
    package: Path,
) -> None:
    assert _prefix_caching_argument(package) is False, (
        f"{package.name} speculates over a recurrent state, so it must"
        f' declare required_arguments={{"{_FLAG}": False}}. With prefix'
        " caching on, its checkpoints are published at the wrong depth."
    )
