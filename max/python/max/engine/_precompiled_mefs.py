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

Artifacts are matched by name rather than by a compile key. A compile key covers
the host CPU target, kernel-package contents and the build configuration, which
two different machines rarely agree on, and a key that fails to match falls back
to compiling silently. Naming an artifact after what it holds removes the key
from the picture, and a graph with no artifact raises instead -- a split that
stops working needs to say so.

The default name is the graph's name and a digest of its input and output types,
so nothing depends on the order the graphs are compiled in. That matters,
because the compiling and the executing run rarely agree on it: the consumer may
run a subset, or reach the same graphs by another path. A graph compiled twice
claims one artifact, which is what a caller wants.

Nothing makes a graph's name unique, though, so one name and signature can
describe two different computations, and exporting a second, different graph
under a taken pair raises. A caller that can tell such graphs apart -- because
it knows which unit of work is compiling, say -- passes its own ``key`` and
names them however it needs to; the collision check still holds it to naming
them apart. Such a caller usually wants ``verify_body`` as well, which checks
the artifact it claims against the graph claiming it.

The signature check is a guard, not a proof: it catches divergences visible in a
graph's input and output types, not every way two graphs can differ. Reuse only
artifacts produced by the same code at the same revision.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable
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


def _signature_digest(signature: dict[str, list[str]]) -> str:
    """Digests a signature short enough to name an artifact with.

    Truncated because it identifies an artifact within one directory rather than
    guarding against a chosen collision.

    Args:
        signature: The signature to digest.

    Returns:
        A truncated hex digest.
    """
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:12]


def _default_key(graph: Graph) -> str:
    """Names ``graph``'s artifact after what it computes.

    The signature digest is what distinguishes two graphs sharing a name, which
    ``Module.compile`` produces routinely: it names every graph after the module
    class, so one model compiled at several shapes yields one name and several
    signatures.

    Args:
        graph: The graph being compiled.

    Returns:
        The artifact's filename.
    """
    digest = _signature_digest(_signature(graph))
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", graph.name)
    return f"{sanitized}-{digest}.mef"


_KERNEL_PATHS = re.compile(r"_kernel_library_paths = \[[^\]]*\]")


def _fingerprint(graph: Graph) -> str:
    """Digests what a graph computes, to tell two same-named graphs apart.

    Compared between graphs built by one exporting process, and between the
    directories separate build actions leave behind, so it has to describe what
    a graph computes and nothing about where it was built. The printed form
    carries no source locations; the kernel-library paths it does carry point
    into the tree that built the graph, so they are dropped.

    Args:
        graph: The graph to digest.

    Returns:
        A hex digest of the graph's body.
    """
    body = _KERNEL_PATHS.sub("_kernel_library_paths = []", repr(graph))
    return hashlib.sha256(body.encode()).hexdigest()


class MissingArtifactError(RuntimeError):
    """No artifact was recorded for a graph a session was asked to import.

    A :class:`RuntimeError` because that is what this path raised before the
    type existed, and callers catch it that way.
    """

    def __init__(
        self,
        graph_name: str,
        key: str,
        signature: dict[str, list[str]],
        directories: tuple[Path, ...],
        recorded: Iterable[str],
    ) -> None:
        self.graph_name = graph_name
        self.key = key
        self.signature = signature
        self.directories = directories
        super().__init__(
            f"no precompiled artifact {key!r} for graph {graph_name!r} with "
            f"signature {signature}, in "
            + ", ".join(str(path) for path in directories)
            + ".\n  precompiled: "
            + (", ".join(sorted(recorded)) or "(nothing)")
            + "\nThe exporting and importing runs must build the same graphs; "
            "config derived from device memory (batch size, for one) has to be "
            "pinned explicitly on both sides."
        )


class ArtifactBodyMismatch(RuntimeError):
    """The recorded artifact holds a different graph than the one importing it.

    The recorded body is the one check that catches a divergence a name and a
    signature cannot, so a store asked to verify it reports rather than
    initializing an artifact of some other computation.
    """

    def __init__(
        self, graph_name: str, key: str, recorded: str, actual: str
    ) -> None:
        self.graph_name = graph_name
        self.key = key
        self.recorded = recorded
        self.actual = actual
        super().__init__(
            f"the artifact recorded as {key!r} describes a different graph "
            f"than {graph_name!r}, which is claiming it: recorded "
            f"{recorded[:12]}, built {actual[:12]}.\nThe two runs diverged in "
            "what they built. Usual causes: constants from an unseeded random "
            "source, graph construction that branches on "
            "is_virtual_device_mode(), and shapes or a batch size derived from "
            "the device's free memory."
        )


@dataclass
class _Entry:
    key: str
    name: str
    signature: dict[str, list[str]]
    # Absent from a v1 manifest, and so from every entry read back from one:
    # it answered a question only the exporting process could ask.
    fingerprint: str = ""


@dataclass
class MefStore:
    """A directory of compiled-graph artifacts, being written or read.

    Construct with :meth:`for_export` or :meth:`for_import` rather than
    directly.
    """

    directories: tuple[Path, ...]
    exporting: bool
    # What to call a graph's artifact, and whether to check the one claimed
    # against the graph claiming it. Both belong to whoever built the store,
    # because only a caller that knows what it is compiling can improve on the
    # defaults.
    key: Callable[[Graph], str | None]
    verify_body: bool
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _entries: dict[str, _Entry] = field(default_factory=dict)
    # Where each artifact actually lives, which is not derivable from the key
    # once the artifacts come from more than one directory.
    _artifacts: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def for_export(
        cls,
        directory: str | Path,
        *,
        key: Callable[[Graph], str] | None = None,
    ) -> MefStore:
        """Returns a store that writes artifacts into ``directory``.

        Args:
            directory: Where to write the artifacts and their manifest. Created
                if it does not exist.
            key: What to name each graph's artifact, in place of its name and
                signature digest. For a caller that can tell apart two graphs
                the default would collide; the importing store has to be built
                with a key that agrees with this one.

        Returns:
            The store to pass as a session's ``export_mefs``.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return cls(
            directories=(path,),
            exporting=True,
            key=key or _default_key,
            verify_body=False,
        )

    @classmethod
    def for_import(
        cls,
        directories: str | Path | Iterable[str | Path],
        *,
        key: Callable[[Graph], str | None] | None = None,
        verify_body: bool = False,
    ) -> MefStore:
        """Returns a store that reads the artifacts in ``directories``.

        Several are accepted because one producer often cannot compile every
        graph: a build action has a time limit, so a large set is split across
        actions, each writing its own directory. Since an artifact is named after
        the graph it holds, the directories need no merging -- their manifests
        union, and a key present in two of them usually describes the same graph
        either way. When it does not, neither artifact can be matched back to a
        graph, so the union raises rather than picking one.

        Args:
            directories: One directory written by an exporting store, or several.
            key: How the exporting store named each artifact, if not the
                default. Returning :obj:`None` from it declines a graph, which
                is then compiled as usual.
            verify_body: Whether to check a claimed artifact against the graph
                claiming it, raising :class:`ArtifactBodyMismatch` when the two
                runs diverged. Off by default because a caller may knowingly
                build different graphs than it recorded -- a pipeline under
                virtual devices does.

        Returns:
            The store to pass as a session's ``precompiled_mefs``.

        Raises:
            ValueError: If no directory is named.
            FileNotFoundError: If one holds no manifest, so was not written by a
                session exporting precompiled MEFs.
            RuntimeError: If two directories hold different graphs under one
                key, which no artifact could tell apart.
        """
        if isinstance(directories, (str, Path)):
            directories = [directories]
        paths = tuple(Path(directory) for directory in directories)
        if not paths:
            raise ValueError("name at least one directory of precompiled MEFs")

        entries: dict[str, _Entry] = {}
        artifacts: dict[str, Path] = {}
        for path in paths:
            manifest = path / _MANIFEST_NAME
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"{path} has no {_MANIFEST_NAME}, so it was not written by "
                    "a session exporting precompiled MEFs"
                )
            for entry in json.loads(manifest.read_text())["graphs"]:
                recorded = _Entry(**entry)
                seen = entries.get(recorded.key)
                if (
                    seen is not None
                    and seen.fingerprint != recorded.fingerprint
                ):
                    raise RuntimeError(
                        f"{artifacts[recorded.key]} and {path / recorded.key}"
                        " describe different graphs, so neither can be matched"
                        f" back to {recorded.name!r}. Nothing makes a graph's"
                        " name unique; give them different names to tell them"
                        " apart."
                    )
                entries.setdefault(recorded.key, recorded)
                artifacts.setdefault(recorded.key, path / recorded.key)

        return cls(
            directories=paths,
            exporting=False,
            key=key or _default_key,
            verify_body=verify_body,
            _entries=entries,
            _artifacts=artifacts,
        )

    def claim_export(self, graph: Graph) -> Path:
        """Records ``graph`` and returns the path to write its artifact to.

        Args:
            graph: The graph about to be compiled.

        Returns:
            Where to export the compiled artifact.

        Raises:
            ValueError: If the store's key declined ``graph``.
            RuntimeError: If a different graph was already exported under the
                same key, which no artifact could tell apart.
        """
        key = self.key(graph)
        if key is None:
            # Declining belongs to the importing half: there, a store with
            # nothing to say lets the graph compile. Here it would drop it.
            raise ValueError(
                f"the exporting store's key declined graph {graph.name!r}; "
                "an exporting store must name every graph it compiles"
            )

        entry = _Entry(
            key=key,
            name=graph.name,
            signature=_signature(graph),
            fingerprint=_fingerprint(graph),
        )
        with self._lock:
            # A graph compiled twice claims one artifact, and the second
            # export rewrites bytes describing the same graph. Two *different*
            # graphs landing on one key is the case no artifact can represent.
            previous = self._entries.get(key)
            if (
                previous is not None
                and previous.fingerprint != entry.fingerprint
            ):
                raise RuntimeError(
                    f"a different graph was already exported as {key!r}"
                    f" ({previous.name!r} against {graph.name!r}), so an"
                    " artifact cannot say which of the two it holds. Nothing"
                    " makes a graph's name unique; give them different names,"
                    " or name their artifacts with a key that tells them apart."
                )
            self._entries[key] = entry
        return self.directories[0] / key

    def claim_import(self, graph: Graph) -> Path | None:
        """Returns the artifact for ``graph``, or :obj:`None` to compile it.

        Args:
            graph: The graph that would otherwise be compiled.

        Returns:
            The artifact to initialize in its place, or :obj:`None` if this
            store has nothing to say about ``graph``, which is then compiled as
            usual.

        Raises:
            MissingArtifactError: If no artifact was precompiled for this graph.
            ArtifactBodyMismatch: If ``verify_body`` and the artifact claimed
                holds a different graph.
            RuntimeError: If ``verify_body`` and the artifact was recorded
                before graph bodies were, so cannot be checked.
        """
        key = self.key(graph)
        if key is None:
            return None

        with self._lock:
            entry = self._entries.get(key)

        if entry is None:
            with self._lock:
                recorded = [
                    recorded.name for recorded in self._entries.values()
                ]
            raise MissingArtifactError(
                graph.name, key, _signature(graph), self.directories, recorded
            )
        if self.verify_body:
            if not entry.fingerprint:
                raise RuntimeError(
                    f"the artifact recorded as {key!r} predates recorded graph"
                    " bodies, so it cannot be checked against the graph"
                    " claiming it. Re-export these artifacts, or build the"
                    " store without verify_body."
                )
            fingerprint = _fingerprint(graph)
            if entry.fingerprint != fingerprint:
                raise ArtifactBodyMismatch(
                    graph.name, key, entry.fingerprint, fingerprint
                )
        return self._artifacts[entry.key]

    def write_manifest(self) -> None:
        """Writes the manifest describing everything exported so far."""
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.key)
        (self.directories[0] / _MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "graphs": [
                        {
                            "key": entry.key,
                            "name": entry.name,
                            "signature": entry.signature,
                            "fingerprint": entry.fingerprint,
                        }
                        for entry in entries
                    ],
                },
                indent=2,
            )
        )
