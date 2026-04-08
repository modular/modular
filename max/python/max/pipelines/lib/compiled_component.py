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

from max.engine import InferenceSession
from max.pipelines.lib.model_manifest import ModelManifest


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
