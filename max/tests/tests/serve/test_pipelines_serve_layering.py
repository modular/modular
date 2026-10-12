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
"""Layering guard: no module under ``max.pipelines`` may import ``max.serve``.

``max.pipelines`` is the offline-inference closure; ``max.serve`` is the
server that sits above it and owns the telemetry stack. The rule is written
down at ``max/python/max/pipelines/request/open_responses.py`` and holds today
with zero hits, which is why every media metric is emitted from
``max.serve.router`` even where the value is produced further down: the
per-architecture image processor and the preprocess cache both live in
``max.pipelines``, and pulling the metric client in beside them would drag
the serve stack into every offline `generate` call and break the dependency
graph.

This reads the source rather than ``sys.modules``: a violation added inside a
function body or a ``TYPE_CHECKING`` block never shows up at runtime in a test
that has already imported half the tree, and the complementary runtime check
already exists (``max/tests/tests/cascade/test_arch_import_layering.py``).
"""

from __future__ import annotations

import ast
import pathlib

import max.pipelines.context

_PIPELINES_ROOT = pathlib.Path(max.pipelines.context.__file__).parent.parent
_SERVE = "max.serve"

# The scan is over whatever of ``max.pipelines`` this test target's deps put
# in runfiles, which is most of it but not provably all. Below this, assume
# the tree is missing and fail rather than pass on an empty walk.
_MIN_SCANNED_FILES = 100


def _serve_imports(path: pathlib.Path) -> list[str]:
    """Every ``max.serve`` import in one file, as ``file:line: statement``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _SERVE or alias.name.startswith(_SERVE + "."):
                    found.append(f"{path}:{node.lineno}: import {alias.name}")
        # A relative import cannot leave max.pipelines, so only absolute ones
        # can reach the serve layer.
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            and (node.module == _SERVE or node.module.startswith(_SERVE + "."))
        ):
            found.append(f"{path}:{node.lineno}: from {node.module} import ...")
    return found


def test_no_pipelines_module_imports_the_serve_layer() -> None:
    offenders: list[str] = []
    scanned = 0
    for path in sorted(_PIPELINES_ROOT.rglob("*.py")):
        scanned += 1
        offenders.extend(_serve_imports(path))

    assert scanned >= _MIN_SCANNED_FILES, (
        f"only {scanned} file(s) under {_PIPELINES_ROOT} were scanned, so a"
        " violation would go unnoticed; the package is not in runfiles"
    )
    assert not offenders, (
        "max.pipelines must not import max.serve -- it is the offline"
        " closure, and the serve telemetry stack cannot be part of it."
        " Move the emission to max.serve and carry the value across the"
        " seam instead. Offenders:\n" + "\n".join(offenders)
    )
