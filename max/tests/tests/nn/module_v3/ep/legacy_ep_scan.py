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
"""Finds the imports of the legacy expert-parallel exchange in Python source.

It reads the source rather than ``sys.modules``, so an import inside a
function or a ``TYPE_CHECKING`` block counts too, and a docstring that names
the legacy exchange does not.
"""

from __future__ import annotations

import ast
import pathlib

_LEGACY_EP_SEGMENTS = "legacy.comm.ep"

_LEGACY_EP_NAMES = frozenset(
    {"EPBatchManager", "EPCommBuffers", "EPCommInitializer"}
)
"""The legacy exchange's public names, wherever they are imported from."""


def _names_legacy_ep(node: ast.Import | ast.ImportFrom) -> bool:
    """Whether ``node`` imports the legacy exchange or one of its names.

    A relative import names the module without ``max.nn``, so the match is
    on the ``legacy.comm.ep`` segments.
    """
    module = getattr(node, "module", None) or ""
    names = [alias.name for alias in node.names]
    modules = [module, *(".".join(filter(None, (module, n))) for n in names)]
    return bool(_LEGACY_EP_NAMES.intersection(names)) or any(
        f".{_LEGACY_EP_SEGMENTS}." in f".{m}." for m in modules
    )


def legacy_ep_imports(source: str, filename: str) -> list[str]:
    """Returns each import of the legacy exchange in ``source``.

    Args:
        source: The Python source to read.
        filename: The name to report each import under.

    Returns:
        One ``filename:line`` entry per import that names the legacy
        exchange's module, a module under it, or one of its public names.
    """
    return [
        f"{filename}:{node.lineno}"
        for node in ast.walk(ast.parse(source, filename=filename))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _names_legacy_ep(node)
    ]


def scan(paths: list[pathlib.Path]) -> list[str]:
    """Returns each import of the legacy exchange in ``paths``."""
    found: list[str] = []
    for path in paths:
        found.extend(
            legacy_ep_imports(path.read_text(encoding="utf-8"), str(path))
        )
    return found
