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
"""FlashInfer FP4 GEMM custom op for loading TVM FFI modules."""

import compiler_internal as compiler
import format
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import CUstream
from memory import Span, stack_allocation
from os import abort
from runtime.asyncrt import DeviceContextPtr
import sys
from sys.ffi import OwnedDLHandle
from tensor import InputTensor, OutputTensor, ManagedTensorSlice
from utils import IndexList

from . import dlpack, tvm_ffi


@fieldwise_init
struct Module:
    var lib: OwnedDLHandle

    def fp4_gemm(
        self,
        mat1: dlpack.DLTensor[dtype = DType.float4_e2m1fn, rank=2],
        mat2: dlpack.DLTensor[dtype = DType.float4_e2m1fn, rank=2],
        mat1_scale: dlpack.DLTensor[dtype = DType.uint8, rank=1],
        mat2_scale: dlpack.DLTensor[dtype = DType.uint8, rank=1],
        global_scale: dlpack.DLTensor[dtype = DType.float32, rank=1],
        out_tensor: dlpack.DLTensor[dtype = DType.bfloat16, rank=2],
        workspace: dlpack.DLTensor[dtype = DType.int8, rank=1],
        tactic: Int = 0,  # auto
    ) -> None:
        safe_call = self.lib.get_function[tvm_ffi.SafeFunction](
            "__tvm_ffi_fp4_gemm"
        )

        # kernel expects packed uint8
        mat1_uint8 = mat1.copy()
        mat1_uint8.data_type = dlpack.DLDataType.from_dtype[DType.uint8]()
        mat1_uint8.shape[-1] *= 2
        mat2_uint8 = mat2.copy()
        mat2_uint8.data_type = dlpack.DLDataType.from_dtype[DType.uint8]()
        mat2_uint8.shape[-1] *= 2

        args: InlineArray[tvm_ffi.TVMFFIAny, 8] = [
            tvm_ffi.TVMFFIAny(mat1_uint8),
            tvm_ffi.TVMFFIAny(mat2_uint8),
            tvm_ffi.TVMFFIAny(mat1_scale),
            tvm_ffi.TVMFFIAny(mat2_scale),
            tvm_ffi.TVMFFIAny(global_scale),
            tvm_ffi.TVMFFIAny(out_tensor),
            tvm_ffi.TVMFFIAny(workspace),
            tvm_ffi.TVMFFIAny(tactic),
        ]

        # Result allocation (will be None, no return for fp4_gemm)
        result = tvm_ffi.TVMFFIAny(0)

        # Call the function
        errno = safe_call(
            UnsafePointer[NoneType, MutAnyOrigin](),  # null unused module
            Pointer[tvm_ffi.TVMFFIAny, MutAnyOrigin](to=args[0]),
            8,  # num_args
            Pointer[tvm_ffi.TVMFFIAny, MutAnyOrigin](to=result),
        )

        if errno != 0:
            error = tvm_ffi.take_latest_error()
            raise Error("FlashInfer fp4_gemm failed: {}".format(error))


# ===-----------------------------------------------------------------------===#
# Custom Op
# ===-----------------------------------------------------------------------===#


@compiler.register("flashinfer_fp4_gemm")
struct FlashInferFP4Gemm[lib_path: StaticString]:
    """Custom op that calls FlashInfer FP4 GEMM via TVM FFI.

    Parameters:
        lib_path: Path to the FlashInfer .so file built by flashinfer.aot.

    Inputs:
        - mat1: [M, K/2] uint8 (packed FP4)
        - mat2: [N, K/2] uint8 (packed FP4)
        - mat1_scale: scale factors for mat1
        - mat2_scale: scale factors for mat2
        - global_scale: [1] float32
        - workspace: workspace buffer

    Output:
        - out: [M, N] bfloat16

    Note: TVM FFI must be loaded by Python (import tvm_ffi and
    tvm_ffi.module.load_module) before this op runs.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_tensor: OutputTensor[dtype = DType.bfloat16, rank=2],
        mat1: InputTensor[dtype = DType.float4_e2m1fn, rank=2],
        mat2: InputTensor[dtype = DType.float4_e2m1fn, rank=2],
        mat1_scale: InputTensor[dtype = DType.uint8, rank=1],
        mat2_scale: InputTensor[dtype = DType.uint8, rank=1],
        global_scale: InputTensor[dtype = DType.float32, rank=1],
        workspace: InputTensor[dtype = DType.int8, rank=1],
        ctx: DeviceContextPtr,
    ):
        """Execute the FP4 GEMM operation by calling FlashInfer."""
        constrained[target == "gpu"]()

        mod = Module(OwnedDLHandle(path=Self.lib_path))

        mod.fp4_gemm(
            dlpack.DLTensor(mat1),
            dlpack.DLTensor(mat2),
            dlpack.DLTensor(mat1_scale),
            dlpack.DLTensor(mat2_scale),
            dlpack.DLTensor(global_scale),
            dlpack.DLTensor(out_tensor),
            dlpack.DLTensor(workspace),
        )
