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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os
import pathlib
from collections.abc import Sequence

import max._core
import max._core.dialects.kgen
import max._core.dialects.m
import max._core.dialects.mo
import max._core.driver
import max._core.dtype
import max._mlir.ir

def load_modular_dialects(arg: max._mlir.ir.DialectRegistry, /) -> None: ...
def array_attr(
    arg0: max._core.driver.Tensor, arg1: max._core.dialects.mo.TensorType, /
) -> max._core.dialects.m.ArrayElementsAttr: ...
def next_operation(
    arg: max._mlir.ir.Operation, /
) -> max._mlir.ir.Operation: ...
def prev_operation(
    arg: max._mlir.ir.Operation, /
) -> max._mlir.ir.Operation: ...
def last_operation(arg: max._mlir.ir.Block, /) -> max._mlir.ir.Operation: ...
def dtype_to_type(arg: max._core.dtype.DType, /) -> max._core.Type: ...
def type_to_dtype(arg: max._core.Type, /) -> max._core.dtype.DType: ...
def frame_loc(
    arg0: max._mlir.ir.Context, arg1: object, /
) -> max._mlir.ir.Location: ...

class Analysis:
    def __init__(
        self, arg0: object, arg1: Sequence[str | os.PathLike], /
    ) -> None: ...
    @property
    def symbol_names(self) -> list[str]: ...
    @property
    def library_paths(self) -> list[pathlib.Path]: ...
    def kernel(self, arg: str, /) -> max._core.dialects.kgen.GeneratorOp: ...
    def verify_custom_op(self, arg: max._mlir.ir.Operation, /) -> None: ...
    def add_path(self, arg: str | os.PathLike, /) -> None: ...
