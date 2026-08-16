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

"""Compile a set of graphs once, then reuse the artifacts by path.

Graph compilation does not need the accelerator it targets, but execution does,
so a caller that compiles where it executes spends scarce accelerator time on
work a CPU could have done. :meth:`~max.engine.InferenceSession.compile` and
:meth:`~max.engine.InferenceSession.init` already split those halves, but a
caller driving something that builds its own graphs -- a pipeline, say -- never
sees them to split them itself.

:class:`MefStore` is what :class:`~max.engine.InferenceSession` consults to do it
on the caller's behalf. A session constructed with ``export_mefs`` writes each
graph it compiles into that directory; one constructed with ``precompiled_mefs``
initializes those artifacts instead of compiling.

Artifacts are matched by position and then verified against the graph's name and
signature, rather than by a compile key. A compile key covers the host CPU
target, kernel-package contents and the build configuration, which two different
machines rarely agree on, and a key that fails to match falls back to compiling
silently. Matching by path removes the key from the picture, and a mismatch
raises instead -- a split that stops working needs to say so.

The signature check is a guard, not a proof: it catches divergences visible in a
graph's input and output types, not every way two graphs can differ. Reuse only
artifacts produced by the same code at the same revision.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.graph import Graph

_MANIFEST_NAME = "manifest.json"


def _signature(graph: Graph) -> dict[str, list[str]]:
    """Describes a graph's inputs and outputs as comparable strings.

    Deliberately not a hash of the graph itself: MLIR bytecode embeds source
    locations, whose absolute paths differ between the producing and consuming
    trees, so a content hash would never match. The signature carries the
    divergences that matter in practice -- a batch size baked into a shape, or an
    input present on one side only.

    Args:
        graph: The graph to describe.

    Returns:
        The graph's input and output types, rendered as strings.
    """
    return {
        "inputs": [str(value.type) for value in graph.inputs],
        "outputs": [str(output) for output in graph.output_types],
    }


@dataclass
class _Entry:
    key: str
    name: str
    signature: dict[str, list[str]]


@dataclass
class MefStore:
    """A directory of compiled-graph artifacts, being written or read.

    Construct with :meth:`for_export` or :meth:`for_import` rather than
    directly. Tracks the position reached in the directory, so a store belongs to
    one session.
    """

    directory: Path
    exporting: bool
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _entries: list[_Entry] = field(default_factory=list)
    _next_index: int = 0

    @classmethod
    def for_export(cls, directory: str | Path) -> MefStore:
        """Returns a store that writes artifacts into ``directory``.

        Args:
            directory: Where to write the artifacts and their manifest. Created
                if it does not exist.

        Returns:
            The store to pass as a session's ``export_mefs``.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return cls(directory=path, exporting=True)

    @classmethod
    def for_import(cls, directory: str | Path) -> MefStore:
        """Returns a store that reads the artifacts in ``directory``.

        Args:
            directory: A directory written by an exporting store.

        Returns:
            The store to pass as a session's ``precompiled_mefs``.

        Raises:
            FileNotFoundError: If the directory holds no manifest, so was not
                written by an exporting session.
        """
        path = Path(directory)
        manifest = path / _MANIFEST_NAME
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{path} has no {_MANIFEST_NAME}, so it was not written by a "
                "session exporting precompiled MEFs"
            )
        entries = [
            _Entry(**entry)
            for entry in json.loads(manifest.read_text())["graphs"]
        ]
        return cls(directory=path, exporting=False, _entries=entries)

    def claim_export(self, graph: Graph) -> Path:
        """Records ``graph`` and returns the path to write its artifact to.

        Args:
            graph: The graph about to be compiled.

        Returns:
            Where to export the compiled artifact.
        """
        with self._lock:
            index = self._next_index
            self._next_index += 1
            entry = _Entry(
                key=f"{index:03d}-{graph.name}.mef",
                name=graph.name,
                signature=_signature(graph),
            )
            self._entries.append(entry)
            return self.directory / entry.key

    def claim_import(self, graph: Graph) -> Path:
        """Returns the artifact for ``graph``, checking it is the right one.

        Args:
            graph: The graph that would otherwise be compiled.

        Returns:
            The artifact to initialize in its place.

        Raises:
            RuntimeError: If this session compiles more graphs than were
                exported, or a graph does not match the artifact recorded in its
                position.
        """
        with self._lock:
            index = self._next_index
            self._next_index += 1
            if index >= len(self._entries):
                raise RuntimeError(
                    f"this session compiled {index + 1} graphs but only "
                    f"{len(self._entries)} were precompiled into "
                    f"{self.directory}; re-export with the same configuration"
                )
            entry = self._entries[index]

        signature = _signature(graph)
        if entry.name != graph.name or entry.signature != signature:
            raise RuntimeError(
                f"graph {index} does not match the precompiled artifact "
                f"{entry.key!r}, so it cannot be reused.\n"
                f"  expected: {entry.name} {entry.signature}\n"
                f"  building: {graph.name} {signature}\n"
                "The exporting and importing runs must build the same graphs; "
                "config derived from device memory (batch size, for one) has to "
                "be pinned explicitly on both sides."
            )
        return self.directory / entry.key

    def write_manifest(self) -> None:
        """Writes the manifest describing everything exported so far."""
        with self._lock:
            entries = list(self._entries)
        (self.directory / _MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "graphs": [
                        {
                            "key": entry.key,
                            "name": entry.name,
                            "signature": entry.signature,
                        }
                        for entry in entries
                    ]
                },
                indent=2,
            )
        )
