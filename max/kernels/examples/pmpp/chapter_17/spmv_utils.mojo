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
from std.collections import List
from std.builtin.device_passable import DevicePassable


struct COOMatrix(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self
    var numNonzeros: Int
    var numRows: Int
    var numCols: Int
    var rowIdx: UnsafePointer[UInt32, MutAnyOrigin]
    var colIdx: UnsafePointer[UInt32, MutAnyOrigin]
    var value: UnsafePointer[Float32, MutAnyOrigin]

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "COOMatrix"

    def __init__(
        out self,
        nnz: Int,
        rows: Int,
        cols: Int,
        r: UnsafePointer[UInt32, MutAnyOrigin],
        c: UnsafePointer[UInt32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
    ):
        self.numNonzeros = nnz
        self.numRows = rows
        self.numCols = cols
        self.rowIdx = r
        self.colIdx = c
        self.value = v


struct CSRMatrix(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self
    var numRows: Int
    var numCols: Int
    var numNonzeros: Int
    var rowPtrs: UnsafePointer[UInt32, MutAnyOrigin]
    var colIdx: UnsafePointer[UInt32, MutAnyOrigin]
    var value: UnsafePointer[Float32, MutAnyOrigin]

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "CSRMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz: Int,
        r: UnsafePointer[UInt32, MutAnyOrigin],
        c: UnsafePointer[UInt32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
    ):
        self.numRows = rows
        self.numCols = cols
        self.numNonzeros = nnz
        self.rowPtrs = r
        self.colIdx = c
        self.value = v


struct ELLMatrix(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self
    var numRows: Int
    var numCols: Int
    var nnzPerRow: Int
    var colIdx: UnsafePointer[UInt32, MutAnyOrigin]
    var value: UnsafePointer[Float32, MutAnyOrigin]

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "ELLMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz_per_row: Int,
        c: UnsafePointer[UInt32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
    ):
        self.numRows = rows
        self.numCols = cols
        self.nnzPerRow = nnz_per_row
        self.colIdx = c
        self.value = v


struct CSCMatrix(DevicePassable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self
    var numRows: Int
    var numCols: Int
    var numNonzeros: Int
    var colPtrs: UnsafePointer[UInt32, MutAnyOrigin]
    var rowIdxs: UnsafePointer[UInt32, MutAnyOrigin]
    var values: UnsafePointer[Float32, MutAnyOrigin]

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "CSCMatrix"

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        nnz: Int,
        c: UnsafePointer[UInt32, MutAnyOrigin],
        r: UnsafePointer[UInt32, MutAnyOrigin],
        v: UnsafePointer[Float32, MutAnyOrigin],
    ):
        self.numRows = rows
        self.numCols = cols
        self.numNonzeros = nnz
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
                values.append(random_float64().cast[DType.float32]())


def verify(
    y_ref: List[Float32],
    d_y: UnsafePointer[Float32, MutAnyOrigin],
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
