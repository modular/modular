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

"""MO IR dump helpers."""

from __future__ import annotations

from pathlib import Path

from max.graph import Graph


def dump_mo_ir(graph: Graph, output_path: Path) -> Path:
    """Write MO-level MLIR IR for a graph to a file.

    Args:
        graph: The MAX graph object (before session.load).
        output_path: Path to write the .mo.mlir file.

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mo_ir = str(graph)
    output_path.write_text(mo_ir)
    return output_path
