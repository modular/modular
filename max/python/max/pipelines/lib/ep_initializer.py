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
"""What a pipeline needs of the expert-parallel exchange's host side."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import BufferType
from max.nn.ep_config import EPConfig

from .graph_specs import _GraphTracer

__all__ = ["EPInitModule", "EPInitializerFactory", "EPInitializerLike"]


class EPInitModule(_GraphTracer, Protocol):
    """Traces the ``ep_init`` graph, which a pipeline compiles as a role.

    :class:`~max.nn.ep.init.EPInitGraph` is one, and so is the legacy
    ``EPInit``.
    """

    def input_types(self) -> list[BufferType]:
        """Returns the graph's input types: each device's atomic counters."""
        ...


class EPInitializerLike(Protocol):
    """Sets up the expert-parallel exchange and hands its buffers to steps.

    A ModuleV2 model's pipeline uses the legacy ``EPCommInitializer``, which
    passes the buffers in the order of
    :func:`~max.nn.ep.init.payload_types`. A ModuleV3 model's starts the
    exchange through
    :meth:`~max.pipelines.lib.registration_runtime.ExpertParallelInputs.initialize`.
    """

    config: EPConfig
    """The expert-parallel configuration. :meth:`ep_init` sets its
    ``node_id``."""

    def ep_init(self, session: InferenceSession) -> None:
        """Compiles and runs the ``ep_init`` graph on this node's devices."""
        ...

    def init_module(self) -> EPInitModule:
        """Returns the ``ep_init`` graph; touches no device."""
        ...

    def run_init(self, init_model: Model) -> None:
        """Runs the compiled graph of :meth:`init_module`."""
        ...

    def model_inputs(self) -> list[Buffer]:
        """Returns the buffers in the standard graph-input order."""
        ...

    def set_eplb_placement(
        self, log2phy: Sequence[Buffer], logcnt: Sequence[Buffer]
    ) -> None:
        """Supplies the per-device EPLB tables :meth:`model_inputs` returns."""
        ...


EPInitializerFactory = Callable[[EPConfig], EPInitializerLike]
"""Makes the initializer for an expert-parallel configuration."""
