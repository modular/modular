# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
import os
from collections.abc import Sequence
from typing import Any, Mapping, overload

import max._core

class FrameworkFormat(enum.Enum):
    max_graph = 0

    torchscript_module = 1

    torchscript_function = 2

    torch_mlir = 3

class InferenceSession:
    def __init__(self, config: dict = {}) -> None: ...
    def compile_from_path(
        self, model_path: str | os.PathLike, config: dict = {}
    ) -> Model: ...
    def compile_from_object(
        self, model: object, format: FrameworkFormat, config: dict = {}
    ) -> Model: ...
    @property
    def stats_report(self) -> str: ...
    def reset_stats_report(self) -> None: ...
    def set_debug_print_options(
        self, style: PrintStyle, precision: int, directory: str
    ) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: bool) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: int) -> None: ...
    @overload
    def set_mojo_define(self, key: str, value: str) -> None: ...
    @property
    def devices(self) -> list[max._core.driver.Device]: ...

class Model:
    def load(self, weights_registry: Mapping[str, Any]) -> None: ...
    @property
    def num_inputs(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def input_metadata(self) -> list[TensorSpec]: ...
    @property
    def output_metadata(self) -> list[TensorSpec]: ...
    @property
    def input_devices(self) -> list[max._core.driver.Device]: ...
    @property
    def output_devices(self) -> list[max._core.driver.Device]: ...
    @property
    def devices(self) -> list[max._core.driver.Device]: ...
    def execute(self, **kwargs) -> dict[str, Any]: ...
    def execute_device_tensors(
        self, *tensors: list[max._core.driver.Tensor | MojoValue]
    ) -> list[max._core.driver.Tensor]: ...

class MojoValue:
    pass

class PrintStyle(enum.Enum):
    COMPACT = 0

    FULL = 1

    BINARY = 2

    BINARY_MAX_CHECKPOINT = 3

    NONE = 4

class TensorData:
    def __init__(
        self, ptr: int, shape: Sequence[int], dtype: max._core.dtype.DType
    ) -> None: ...

class TensorSpec:
    def __init__(
        self,
        shape: Sequence[int | None] | None,
        dtype: max._core.dtype.DType,
        name: str,
    ) -> None: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg: tuple, /) -> None: ...
    @property
    def shape(self) -> list[int | None] | None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def name(self) -> str: ...

class TorchInputSpec:
    def __init__(
        self,
        shape: Sequence[int | str | None] | None,
        dtype: max._core.dtype.DType,
        device: str = "",
    ) -> None: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg: tuple, /) -> None: ...
    @property
    def shape(self) -> list[int | str | None] | None: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def device(self) -> str: ...
