# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os
import pathlib
from collections.abc import Sequence

import max._core
import max._core.dialects.kgen
import max._core.dtype
from numpy.typing import ArrayLike

# isort: off
from max._mlir._mlir_libs._mlir import MlirOperation  # type: ignore[attr-defined]
from max._mlir._mlir_libs._mlir import MlirContext  # type: ignore[attr-defined]
from max._mlir._mlir_libs._mlir import MlirAttribute  # type: ignore[attr-defined]
from max._mlir._mlir_libs._mlir import MlirType  # type: ignore[attr-defined]
from max._mlir._mlir_libs._mlir import MlirLocation  # type: ignore[attr-defined]
from max._mlir._mlir_libs._mlir import MlirDialectRegistry  # type: ignore[attr-defined]
# isort: on

def load_modular_dialects(arg: MlirDialectRegistry, /) -> None: ...
def array_attr(
    arg0: str, arg1: ArrayLike, arg2: MlirType, /
) -> MlirAttribute: ...
def dtype_to_type(arg: max._core.dtype.DType, /) -> max._core.Type: ...
def type_to_dtype(arg: max._core.Type, /) -> max._core.dtype.DType: ...
def frame_loc(arg0: MlirContext, arg1: object, /) -> MlirLocation: ...
def get_frame(arg: MlirLocation, /) -> object: ...

class Analysis:
    def __init__(
        self, arg0: object, arg1: Sequence[str | os.PathLike], /
    ) -> None: ...
    @property
    def symbol_names(self) -> list[str]: ...
    @property
    def library_paths(self) -> list[pathlib.Path]: ...
    def kernel(self, arg: str, /) -> max._core.dialects.kgen.GeneratorOp: ...
    def verify_custom_op(self, arg: MlirOperation, /) -> None: ...
    def add_path(self, arg: str | os.PathLike, /) -> None: ...
