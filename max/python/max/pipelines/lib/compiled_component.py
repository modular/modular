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

"""Base class for self-contained compiled graph components."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from max._core.profiler import Trace, is_profiling_enabled
from max.engine import InferenceSession, Model
from max.graph import Graph, Module
from max.pipelines.lib.model_manifest import ModelManifest

logger = logging.getLogger("max.pipelines")


class CompiledComponent:
    """Base for self-contained compiled graph components.

    Each subclass encapsulates the full lifecycle of one or more compiled
    graphs: config extraction from the manifest, weight loading, Module
    construction, graph building, and runtime execution.

    The base class stores only the shared state (``manifest`` and
    ``session``).  Subclasses own everything else — graph construction,
    weight adaptation, and execution — because the structure varies
    widely across components (single module forward pass vs. fused
    multi-module + ops).

    Subclasses define their own typed ``__call__`` with explicit
    input/output signatures.  There is no shared execution interface.

    Compilation mode:

    * **Eager (default)**: each component calls :meth:`_load_graph`
      during ``__init__``, which immediately invokes ``session.load``
      and logs ``Compiling/Compiled X graph in N.NNs``.
    * **Deferred** (when constructed with ``graphs_module=...``): the
      component adds its :class:`~max.graph.Graph` to the shared
      :class:`~max.graph.Module` and records its weights for the parent
      executor.  The executor then issues a single ``session.load_all``
      across all registered graphs and calls
      :meth:`_attach_compiled_model` on each component to install the
      compiled :class:`Model`.
    """

    # Subclasses may refine ``_model`` to a more specific type (e.g.
    # ``BlockLevelModel | None`` in WAN's transformer).  Typed as ``Any``
    # on the base so that subclass annotations and the assignments in
    # ``_load_graph`` / ``_attach_compiled_model`` below don't conflict.
    _model: Any

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        *,
        graphs_module: Module | None = None,
    ) -> None:
        self._manifest = manifest
        self._session = session
        self._graphs_module = graphs_module
        # Populated by _load_graph in deferred mode; consumed by the
        # parent executor after session.load_all returns.
        self._pending_graph_name: str | None = None
        self._pending_weights_registry: dict[str, Any] = {}

    def _load_graph(self, graph: Graph, **kwargs: Any) -> None:
        """Compile a graph via the session (eager) or defer for ``load_all``.

        In eager mode wraps ``session.load`` in a ``<ClassName>.compile``
        profiling span, logs the elapsed time, and stores the compiled
        :class:`Model` on ``self._model``.  In deferred mode records the
        graph's ``sym_name`` and ``weights_registry`` on the instance;
        the parent executor merges these across components and calls
        ``session.load_all`` once, then installs each compiled
        :class:`Model` via :meth:`_attach_compiled_model`.
        """
        name = type(self).__name__
        if self._graphs_module is None:
            logger.info("Compiling %s graph...", name)
            t0 = time.perf_counter()
            if is_profiling_enabled():
                with Trace(f"{name}.compile"):
                    self._model = self._session.load(graph, **kwargs)
            else:
                self._model = self._session.load(graph, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info("Compiled %s graph in %.2fs", name, elapsed)
            return

        self._pending_graph_name = graph.name
        self._pending_weights_registry = dict(
            kwargs.get("weights_registry") or {}
        )

    def _attach_compiled_model(self, models: Mapping[str, Model]) -> None:
        """Install this component's compiled :class:`Model`.

        Resolved from the ``session.load_all`` result; called by the
        parent executor after the batched compile finishes.
        """
        if self._pending_graph_name is None:
            raise RuntimeError(
                f"{type(self).__name__}: no deferred graph was registered; "
                "_attach_compiled_model is only valid when constructed "
                "with graphs_module=."
            )
        self._model = models[self._pending_graph_name]

    @property
    def _pending_weights(self) -> dict[str, Any]:
        """Weights registered for the batched compile.  Empty in eager mode."""
        return self._pending_weights_registry
