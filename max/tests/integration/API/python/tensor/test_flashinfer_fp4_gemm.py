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
"""Test FlashInfer FP4 GEMM custom op."""

import os
from pathlib import Path

import torch
import tvm_ffi
from flashinfer.aot import gen_gemm_sm100_module_cutlass_fp4
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.tensor import Tensor, TensorType


def _setup_ninja_path() -> None:
    """Add ninja binary to PATH for FlashInfer JIT compilation."""
    import ninja

    if not (ninja_bin_dir := ninja.BIN_DIR):
        # In Bazel pycross_wheel_library, bin is at ../../bin relative to
        # the package location.
        ninja_bin_dir = str(Path(ninja.__file__).parent.parent.parent / "bin")

    os.environ["PATH"] = f"{ninja_bin_dir}{os.pathsep}{os.environ['PATH']}"


def test_flashinfer_fp4_gemm_custom_op() -> None:
    """Test the FlashInfer FP4 GEMM Mojo custom op."""
    # Get the custom ops path
    ops_path = Path(os.environ["FLASHINFER_FP4_GEMM_OPS_PATH"])

    # Build the FlashInfer FP4 GEMM kernel using AOT compilation
    _setup_ninja_path()
    spec = gen_gemm_sm100_module_cutlass_fp4()
    spec.build(verbose=True)
    lib_path = str(spec.jit_library_path)

    M, N, K = 128, 256, 512

    def tensor(shape: list[int], dtype: DType) -> Tensor:
        return Tensor.from_dlpack(
            Buffer(shape=shape, dtype=dtype, device=Accelerator())
        )

    # Create input tensors on GPU
    mat1 = tensor([M, K], dtype=DType.float4_e2m1fn)
    mat2 = tensor([N, K], dtype=DType.float4_e2m1fn)

    assert mat1.device.api == "cuda", "Expected CUDA device"

    def align_up(x: int, align: int) -> int:
        return ((x + align - 1) // align) * align

    def scale_shape(a: int, b: int) -> int:
        return align_up(a, 128) * align_up(b // 16, 4)

    mat1_scale = tensor([scale_shape(M, K)], dtype=DType.uint8)
    mat2_scale = tensor([scale_shape(N, K)], dtype=DType.uint8)
    global_scale = tensor([1], dtype=DType.float32)
    workspace = tensor([1024 * 1024], dtype=DType.int8)

    out_type = TensorType(
        shape=[M, N], dtype=DType.bfloat16, device=Accelerator()
    )

    # Set up TVM FFI stream
    dev = tvm_ffi.device("cuda:0")
    stream = torch.cuda.current_stream().cuda_stream
    print(f"TVM FFI device: {dev}, stream: {stream}")

    with tvm_ffi.use_raw_stream(dev, stream):
        # Call the custom op with the library path as a parameter
        result = F.custom(
            "flashinfer_fp4_gemm",
            device=mat1.device,
            values=[
                mat1,
                mat2,
                mat1_scale,
                mat2_scale,
                global_scale,
                workspace,
            ],
            out_types=[out_type],
            parameters={"lib_path": lib_path},
            custom_extensions=[ops_path],
        )[0]

    # Basic sanity checks
    assert result.real
    assert result.type.shape == [M, N]
    assert result.type.dtype == DType.bfloat16
