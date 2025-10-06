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

from __future__ import annotations

import dataclasses
import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from max.nn._identity import IdentityMap
from max.nn.layer import Layer


@dataclass
class LayerInfo:
    layer_name: str
    call_count: int = 0


class BasePrintHook(ABC):
    """Base hook for printing values.

    This class defines a `__call__` method that prints the inputs and outputs.
    Only layers that have been saved with `hook.add_layer` will be printed.

    Abstract methods:
    - print_value(name, value) -> bool: Override this method to print the value,
        and return whether the print was successful.

    Properties:
    - export_path: The base path to write values into (can be None). The full
        exported path is "{export_path}/{step}".

    Methods:
    - __call__(layer, args, kwargs, outputs): Calls `self.print_value`.
    - add_layer(layer, name): Stores the name of a layer.
    - step(): Can be called to increment the step and update the export path.
    - remove(): To be used to remove the hook and run some cleanup steps.
    - summarize(): Summarize the total number of tensors printed at each step.
    """

    def __init__(self, export_path: str | None = None) -> None:
        self._known_layers = IdentityMap[Layer, LayerInfo]()
        self._export_path = export_path
        self._current_step = 0
        # Keep a counter for creating unique layer names.
        self._layer_counter: dict[str, int] = Counter()

        # Maps step number -> [list of printed tensors]
        self._recorded_prints: dict[int, list[str]] = {}

    def add_layer(self, layer: Any, name: str) -> None:
        self._known_layers[layer] = LayerInfo(name)

    @property
    def export_path(self) -> str | None:
        if self._export_path is None:
            return None
        return os.path.join(self._export_path, str(self._current_step))

    def step(self) -> None:
        self._current_step += 1

        # Update export path.
        if export_path := self.export_path:
            os.makedirs(export_path, exist_ok=True)

            # Reset layer call counts.
            for info in self._known_layers.values():
                info.call_count = 0

        self.write_keys_file()

    def __call__(
        self,
        layer: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        outputs: Any,
    ) -> None:
        """Print all TensorValues."""
        if layer not in self._known_layers:
            # If layer is not yet named, use the class name.
            layer_cls = type(layer).__name__
            self._layer_counter[layer_cls] += 1
            layer_name = layer_cls
            if self._layer_counter[layer_cls] > 1:
                layer_name = f"{layer_name}{self._layer_counter[layer_cls]}"
            self._known_layers[layer] = LayerInfo(layer_name)

        # Update call count and get layer name.
        info = self._known_layers[layer]
        info.call_count += 1
        if info.call_count > 1:
            debug_name = f"{info.layer_name}({info.call_count})"
        else:
            debug_name = info.layer_name

        # Print input args and kwargs
        for n, arg in enumerate(args):
            self.print_and_record(f"{debug_name}-input_{n}", arg)
        for key, value in kwargs.items():
            self.print_and_record(f"{debug_name}-input_{key}", value)

        # Print outputs.
        if self.print_and_record(f"{debug_name}-output", outputs):
            pass
        elif isinstance(outputs, list | tuple):
            for n, out in enumerate(outputs):
                self.print_and_record(f"{debug_name}-output_{n}", out)
        elif dataclasses.is_dataclass(outputs):
            for field in dataclasses.fields(outputs):
                self.print_and_record(
                    f"{debug_name}-output_{field.name}",
                    getattr(outputs, field.name),
                )
        else:
            print(
                f"Was not able to write outputs from {debug_name} (output type"
                f" was {type(outputs)})"
            )
        self.write_keys_file()

    def print_and_record(self, name: str, value: Any) -> bool:
        """Runs self.print_value and records the tensor if printed."""
        print_success = self.print_value(name, value)
        if print_success:
            if self._current_step not in self._recorded_prints:
                self._recorded_prints[self._current_step] = []
            self._recorded_prints[self._current_step].append(name)

        return print_success

    def write_keys_file(self) -> None:
        """Write the list of tensor file names, in the order of execution, to tensor_names.txt ."""

        if self.export_path and self._current_step in self._recorded_prints:
            keys_file = os.path.join(self.export_path, "tensor_names.txt")

            with open(keys_file, "w") as f:
                for name in self._recorded_prints[self._current_step]:
                    f.write(f"{name}\n")

    @abstractmethod
    def print_value(self, name: str, value: Any) -> bool:
        """Prints a value, and returns whether the print is successful."""
        raise NotImplementedError

    def summarize(self) -> None:
        action = "Printed"
        if self.export_path:
            action = "Saved"

        tensors_printed = False
        for step, tensors in self._recorded_prints.items():
            print(f"{action} {len(tensors)} tensors for step {step}.")
            tensors_printed = tensors_printed or bool(len(tensors))

        if not tensors_printed:
            print("No tensors exported.")
        elif self._export_path:
            print(f"Tensors exported to {self._export_path}")

    def remove(self) -> None:
        # Clean up export_path if it's empty.
        if (export_path := self.export_path) and not os.listdir(export_path):
            os.rmdir(export_path)
