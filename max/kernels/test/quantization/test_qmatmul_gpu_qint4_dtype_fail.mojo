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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major
from linalg.utils import no_epilogue_fn
from quantization.qmatmul_gpu import matmul_gpu_qint4


def main() raises:
    var c_bf16_data = Array[BFloat16, 4](fill=0)
    var c_f32_data = Array[Float32, 4](fill=0)
    var a_bf16_data = Array[BFloat16, 4](fill=0)
    var a_f32_data = Array[Float32, 4](fill=0)
    var b_data = Array[UInt8, 4](fill=0)

    var c_bf16 = TileTensor(c_bf16_data, row_major[2, 2]())
    var c_f32 = TileTensor(c_f32_data, row_major[2, 2]())
    var a_bf16 = TileTensor(a_bf16_data, row_major[2, 2]()).as_imm()
    var a_f32 = TileTensor(a_f32_data, row_major[2, 2]()).as_imm()
    var b = TileTensor(b_data, row_major[2, 2]()).as_imm()

    # CHECK: [[@LINE+3]]:5: error: no matching function in call to 'matmul_gpu_qint4'
    # CHECK: note: candidate not viable: violated constraint
    # CHECK-NEXT: def matmul_gpu_qint4[c_type: DType, a_type: DType, //,
    matmul_gpu_qint4[32, "gpu"](c_bf16, a_f32, b)

    # CHECK: [[@LINE+3]]:5: error: no matching function in call to 'matmul_gpu_qint4'
    # CHECK: note: candidate not viable: violated constraint
    # CHECK-NEXT: def matmul_gpu_qint4[c_type: DType, a_type: DType, //,
    matmul_gpu_qint4[32, "gpu"](c_f32, a_bf16, b)

    # CHECK: [[@LINE+3]]:5: error: no matching function in call to 'matmul_gpu_qint4'
    # CHECK: note: candidate not viable: violated constraint
    # CHECK-NEXT: def matmul_gpu_qint4[c_type: DType, a_type: DType, EpilogueFnType:
    matmul_gpu_qint4[32, "gpu"](c_bf16, a_f32, b, no_epilogue_fn)

    # CHECK: [[@LINE+3]]:5: error: no matching function in call to 'matmul_gpu_qint4'
    # CHECK: note: candidate not viable: violated constraint
    # CHECK-NEXT: def matmul_gpu_qint4[c_type: DType, a_type: DType, EpilogueFnType:
    matmul_gpu_qint4[32, "gpu"](c_f32, a_bf16, b, no_epilogue_fn)
