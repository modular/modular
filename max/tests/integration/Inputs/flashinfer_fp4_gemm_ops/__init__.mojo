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
import std.format
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUstream
from std.memory import Span, stack_allocation
from std.os import abort
from std.runtime.asyncrt import DeviceContextPtr
from std.ffi import OwnedDLHandle
from tensor import InputTensor, OutputTensor, ManagedTensorSlice
from std.utils import IndexList

from .dlpack import DLTensor
from .tvm_ffi import SafeFunction, TVMFFIAny, take_latest_error


@fieldwise_init
struct Module:
    var lib: OwnedDLHandle

    def fp4_gemm(
        self,
        mat1: DLTensor[dtype=DType.uint8, rank=2],
        mat2: DLTensor[dtype=DType.uint8, rank=2],
        mat1_scale: DLTensor[dtype=DType.uint8, rank=1],
        mat2_scale: DLTensor[dtype=DType.uint8, rank=1],
        global_scale: DLTensor[dtype=DType.float32, rank=1],
        out_tensor: DLTensor[dtype=DType.bfloat16, rank=2],
        workspace: DLTensor[dtype=DType.int8, rank=1],
        tactic: Int = 0,  # auto
    ) -> None:
        safe_call = self.lib.get_function[SafeFunction]("__tvm_ffi_fp4_gemm")

        # `def` params are already mutable local copies, and
        # `DLTensor.__copyinit__` (used at the call site) already fixed
        # up self-referential pointers + nulled strides for contiguous
        # tensors.  So we can take their addresses directly.
        args: InlineArray[TVMFFIAny, 8] = [
            TVMFFIAny(UnsafePointer(to=mat1)),
            TVMFFIAny(UnsafePointer(to=mat2)),
            TVMFFIAny(UnsafePointer(to=mat1_scale)),
            TVMFFIAny(UnsafePointer(to=mat2_scale)),
            TVMFFIAny(UnsafePointer(to=global_scale)),
            TVMFFIAny(UnsafePointer(to=out_tensor)),
            TVMFFIAny(UnsafePointer(to=workspace)),
            TVMFFIAny(tactic),
        ]

        result = TVMFFIAny(0)

        errno = safe_call(
            UnsafePointer[NoneType, MutAnyOrigin](),  # null unused module
            Pointer[TVMFFIAny, MutAnyOrigin](to=args[0]),
            8,  # num_args
            Pointer[TVMFFIAny, MutAnyOrigin](to=result),
        )

        if errno != 0:
            error = take_latest_error()
            raise Error("FlashInfer fp4_gemm failed: {}".format(error))


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

    Note: TVM FFI must be loaded by Python (`import tvm_ffi` and
    `tvm_ffi.module.load_module`) before this op runs.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_tensor: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        mat1: InputTensor[dtype=DType.uint8, rank=2, ...],
        mat2: InputTensor[dtype=DType.uint8, rank=2, ...],
        mat1_scale: InputTensor[dtype=DType.uint8, rank=1, ...],
        mat2_scale: InputTensor[dtype=DType.uint8, rank=1, ...],
        global_scale: InputTensor[dtype=DType.float32, rank=1, ...],
        workspace: InputTensor[dtype=DType.int8, rank=1, ...],
        ctx: DeviceContextPtr,
    ):
        """Execute the FP4 GEMM operation by calling FlashInfer."""
        comptime assert [target == "gpu"]

        mod = Module(OwnedDLHandle(path=Self.lib_path))

        mod.fp4_gemm(
            DLTensor(mat1),
            DLTensor(mat2),
            DLTensor(mat1_scale),
            DLTensor(mat2_scale),
            DLTensor(global_scale),
            DLTensor(out_tensor),
            DLTensor(workspace),
        )
