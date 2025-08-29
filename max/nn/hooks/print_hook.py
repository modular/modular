# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Print hook for MAX Pipeline models."""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Generator
from typing import Any, Optional

from max.graph import TensorValue
from max.nn._identity import IdentitySet
from max.nn.layer import Layer, add_layer_hook, clear_hooks

from .base_print_hook import BasePrintHook

logger = logging.getLogger("max.pipelines")


class PrintHook(BasePrintHook):
    """Hook that prints/saves layer tensor inputs and outputs.

    This class must be initialized added before the graph is built so the
    print ops can be added to the graph.
    """

    def __init__(self, export_path: Optional[str] = None) -> None:
        super().__init__(export_path=export_path)
        add_layer_hook(self)
        if export_path is not None:
            logger.warning(
                "Export path is currently not supported. Values will be printed"
                " to stdout with COMPACT format."
            )

    def name_layers(self, model: Layer) -> None:
        """Create names for all layers in the model based on nested attributes."""
        for layer, name in _walk_layers(model):
            self.add_layer(layer, name)

    @property
    def export_path(self) -> Optional[str]:
        if self._export_path is None:
            return None
        return os.path.join(self._export_path, str(self._current_step))

    def print_value(self, name: str, value: Any) -> bool:
        if isinstance(value, TensorValue):
            value.print(name)
            return True
        return False

    def remove(self) -> None:
        super().remove()
        clear_hooks()  # TODO: Add individual hook remover.

    def __del__(self) -> None:
        self.summarize()


_SUPPORTED_TYPES = (Layer, list, tuple)


def _walk_layers(model: Layer) -> Generator[tuple[Layer, str], None, None]:
    """Walks through model and yields all layers with generated names."""
    seen = IdentitySet[Layer]()
    seen.add(model)
    queue: deque[tuple[Any, str]] = deque([(model, "model")])

    while queue:
        obj, name = queue.popleft()
        if isinstance(obj, Layer):
            yield obj, name
            for k, v in obj.__dict__.items():
                if v not in seen or isinstance(v, _SUPPORTED_TYPES):
                    queue.append((v, f"{name}.{k}"))
        elif isinstance(obj, (list, tuple)):
            for n, v in enumerate(obj):
                if v not in seen or isinstance(v, _SUPPORTED_TYPES):
                    queue.append((v, f"{name}.{n}"))
