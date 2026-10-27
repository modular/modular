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
"""Input batching for the ModuleV3 Inkling pipeline model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import Buffer
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime

from ..inkling.batch_processor import InklingBatchProcessor
from ..inkling.model_config import InklingConfig


class InklingModuleV3BatchProcessor(InklingBatchProcessor):
    """The V2 batching, fed to compiled ModuleV3 callables.

    A compiled ModuleV3 graph supplies its own collective signal buffers, so
    the inputs carry none.
    """

    def __init__(
        self, config: InklingConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._signal_buffers = []
        self._vision_callable: Callable[..., Any] | None = None

    def bind_vision_callable(self, vision_model: Callable[..., Any]) -> None:
        """Hands over the compiled vision tower."""
        self._vision_callable = vision_model

    def _encode_images(self, patches: Buffer) -> Buffer:
        assert self._vision_callable is not None
        embeddings = self._vision_callable(patches).driver_tensor
        assert isinstance(embeddings, Buffer)
        return embeddings
