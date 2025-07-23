# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Print hook for torch models."""

from __future__ import annotations

import os
from typing import Any

import torch
from max.nn.hooks.base_print_hook import BasePrintHook


class TorchPrintHook(BasePrintHook):
    """A torch-compatible print hook."""

    _handle: torch.utils.hooks.RemovableHandle
    """A handle used to remove the forward hook registered by this class."""

    def __init__(self, export_path: str | None = None) -> None:
        super().__init__(export_path)
        self._handle = torch.nn.modules.module.register_module_forward_hook(
            self
        )

        if export_path := self.export_path:
            os.makedirs(export_path, exist_ok=True)

    def name_layers(self, model: torch.nn.Module) -> None:
        """Create names for all layers in the model."""
        for module_name, module in model.named_modules():
            name = f"model.{module_name}" if module_name else "model"
            self.add_layer(module, name)

    @property
    def export_path(self) -> str | None:
        if self._export_path is None:
            return None
        return os.path.join(self._export_path, str(self._current_step))

    def __call__(self, module, args, outputs) -> None:  # type: ignore  # noqa: ANN001
        super().__call__(module, args, kwargs={}, outputs=outputs)

    def print_value(self, name: str, value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if export_path := self.export_path:
                full_path = f"{export_path}/{name}.pt"
                torch.save(value, full_path)
            else:
                print(name, "=", value, value.shape)
            return True
        return False

    def remove(self) -> None:
        super().remove()

        if self._handle:
            self._handle.remove()
