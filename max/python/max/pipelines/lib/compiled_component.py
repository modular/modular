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
from typing import Any

from max._core.profiler import Trace, is_profiling_enabled
from max.engine import InferenceSession, Model
from max.graph import Graph
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
    """

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        self._manifest = manifest
        self._session = session

    def _load_graph(self, graph: Graph, **kwargs: Any) -> Model:
        """Compile a graph via the session, traced under ``max.profiler``.

        Wraps ``session.load()`` in a profiling span named
        ``<ClassName>.compile`` so graph compilation time is visible
        in ``max.profiler`` traces.
        """
        name = type(self).__name__
        logger.info("Compiling %s graph...", name)
        t0 = time.perf_counter()
        if is_profiling_enabled():
            with Trace(f"{name}.compile"):
                model = self._session.load(graph, **kwargs)
        else:
            model = self._session.load(graph, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("Compiled %s graph in %.2fs", name, elapsed)
        return model
