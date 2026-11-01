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

from std.random import random_float64, seed
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


comptime DeviceCOOMatrix = COOMatrix[
    ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin
]


struct COOMatrix[
    row_idx_origin: ImmOrigin,
    col_idx_origin: ImmOrigin,
    value_origin: ImmOrigin,
](DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = DeviceCOOMatrix
    var numNonzeros: Int32
    var numRows: Int32
    var numCols: Int32
    var rowIdx: ImmPointer[UInt32, Self.row_idx_origin]
    var colIdx: ImmPointer[UInt32, Self.col_idx_origin]
    var value: ImmPointer[Float32, Self.value_origin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "COOMatrix"

    def __init__(
        out self,
        nnz: Int,
        rows: Int,
        cols: Int,
        r: ImmPointer[UInt32, Self.row_idx_origin],
        c: ImmPointer[UInt32, Self.col_idx_origin],
        v: ImmPointer[Float32, Self.value_origin],
    ):
        self.numNonzeros = Int32(nnz)
        self.numRows = Int32(rows)
        self.numCols = Int32(cols)
        self.rowIdx = r
        self.colIdx = c
        self.value = v


comptime DeviceCSRMatrix = CSRMatrix[
    ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin
]


struct CSRMatrix[
    row_ptrs_origin: ImmOrigin,
    col_idx_origin: ImmOrigin,
    value_origin: ImmOrigin,
](DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = DeviceCSRMatrix
    var numRows: Int32
    var numCols: Int32
    var numNonzeros: Int32
    var rowPtrs: ImmPointer[UInt32, Self.row_ptrs_origin]
    var colIdx: ImmPointer[UInt32, Self.col_idx_origin]
    var value: ImmPointer[Float32, Self.value_origin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CSRMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz: Int,
        r: ImmPointer[UInt32, Self.row_ptrs_origin],
        c: ImmPointer[UInt32, Self.col_idx_origin],
        v: ImmPointer[Float32, Self.value_origin],
    ):
        self.numRows = Int32(rows)
        self.numCols = Int32(cols)
        self.numNonzeros = Int32(nnz)
        self.rowPtrs = r
        self.colIdx = c
        self.value = v


comptime DeviceELLMatrix = ELLMatrix[ImmUntrackedOrigin, ImmUntrackedOrigin]


struct ELLMatrix[col_idx_origin: ImmOrigin, value_origin: ImmOrigin](
    DevicePassable, TrivialRegisterPassable
):
    comptime device_type: AnyType = DeviceELLMatrix
    var numRows: Int32
    var numCols: Int32
    var nnzPerRow: Int32
    var colIdx: ImmPointer[UInt32, Self.col_idx_origin]
    var value: ImmPointer[Float32, Self.value_origin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ELLMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz_per_row: Int,
        c: ImmPointer[UInt32, Self.col_idx_origin],
        v: ImmPointer[Float32, Self.value_origin],
    ):
        self.numRows = Int32(rows)
        self.numCols = Int32(cols)
        self.nnzPerRow = Int32(nnz_per_row)
        self.colIdx = c
        self.value = v


comptime DeviceCSCMatrix = CSCMatrix[
    ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin
]


struct CSCMatrix[
    col_ptrs_origin: ImmOrigin,
    row_idxs_origin: ImmOrigin,
    values_origin: ImmOrigin,
](DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = DeviceCSCMatrix
    var numRows: Int32
    var numCols: Int32
    var numNonzeros: Int32
    var colPtrs: ImmPointer[UInt32, Self.col_ptrs_origin]
    var rowIdxs: ImmPointer[UInt32, Self.row_idxs_origin]
    var values: ImmPointer[Float32, Self.values_origin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CSCMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz: Int,
        c: ImmPointer[UInt32, Self.col_ptrs_origin],
        r: ImmPointer[UInt32, Self.row_idxs_origin],
        v: ImmPointer[Float32, Self.values_origin],
    ):
        self.numRows = Int32(rows)
        self.numCols = Int32(cols)
        self.numNonzeros = Int32(nnz)
        self.colPtrs = c
        self.rowIdxs = r
        self.values = v


def spmv_cpu(
    rows: Int,
    cols: Int,
    row_idx: List[UInt32],
    col_idx: List[UInt32],
    values: List[Float32],
    x: List[Float32],
    mut y: List[Float32],
):
    for i in range(len(y)):
        y[i] = 0.0
    for i in range(len(values)):
        var r = Int(row_idx[i])
        var c = Int(col_idx[i])
        y[r] += values[i] * x[c]


def generate_sparse_matrix(
    rows: Int,
    cols: Int,
    sparsity: Float32,
    mut row_idx: List[UInt32],
    mut col_idx: List[UInt32],
    mut values: List[Float32],
):
    seed(42)
    for r in range(rows):
        for c in range(cols):
            if random_float64() > Float64(sparsity):
                row_idx.append(UInt32(r))
                col_idx.append(UInt32(c))
                values.append(random_float64().cast[.float32]())


def verify(
    y_ref: List[Float32],
    d_y: ImmPointer[Float32, _],
    rows: Int,
) -> Bool:
    var correct = True
    for i in range(rows):
        var val_gpu = d_y[i]
        var diff = abs(val_gpu - y_ref[i])
        if diff > 1e-3:
            print("Mismatch at row", i, ": ref", y_ref[i], "!= gpu", val_gpu)
            correct = False
            if i > 10:
                break
    return correct
