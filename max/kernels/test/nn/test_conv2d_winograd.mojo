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

from math import isclose
from random import rand

from buffer import DimList, NDBuffer
from nn.conv import Naive2dConvolution

from utils.index import Index


@always_inline
fn matmul[
    dtype: DType, //, N: Int, K: Int, M: Int, transpose_b: Bool
](
    C: NDBuffer[mut=True, dtype, 2, _, _],
    A: NDBuffer[dtype, 2, _, _],
    B: NDBuffer[dtype, 2, _, _],
):
    # TODO: Add constrained[]?
    @parameter
    if transpose_b:
        for i in range(N):
            for j in range(K):
                var sum = Scalar[dtype](0)
                for k in range(M):
                    sum += A[i, k] * B[j, k]
                C[i, j] = sum
    else:
        for i in range(N):
            for j in range(K):
                var sum = Scalar[dtype](0)
                for k in range(M):
                    sum += A[i, k] * B[k, j]
                C[i, j] = sum


# TODO: Less magic numbers for dimensions, use variables
# TODO: This is technically correlation, not convolution. Clarify this.
# TODO: Decide if B, G, and A matrices should be transposed
# TODO: B,G,A can be static
# 12-12-2024: Initial naive version
fn winograd_2d_convolution_3x3[
    dtype: DType
](
    signal: NDBuffer[dtype, 2, _, _],
    kernel: NDBuffer[
        dtype, 2, _, _
    ],  # must be 3x3, let's constrained[]() somehow. Or parameter
    output: NDBuffer[mut=True, dtype, 2, _, _],
):
    # Winograd transformation matrices as stack-allocated NDBuffers
    # fmt: off
    var b_stack = InlineArray[Scalar[dtype], 16](uninitialized=True)
    var B = NDBuffer[dtype, 2,_, DimList((4, 4))](b_stack)
    B[0,0] = 1.0; B[0,1] =  0.0; B[0,2] = -1.0; B[0,3] =  0.0
    B[1,0] = 0.0; B[1,1] =  1.0; B[1,2] =  1.0; B[1,3] =  0.0
    B[2,0] = 0.0; B[2,1] = -1.0; B[2,2] =  1.0; B[2,3] =  0.0
    B[3,0] = 0.0; B[3,1] =  1.0; B[3,2] =  0.0; B[3,3] = -1.0

    var g_stack = InlineArray[Scalar[dtype], 12](uninitialized=True)
    var G = NDBuffer[dtype, 2, _,DimList((4, 3))](g_stack)
    G[0,0] = 1.0; G[0,1] =  0.0; G[0,2] = 0.0
    G[1,0] = 0.5; G[1,1] =  0.5; G[1,2] = 0.5
    G[2,0] = 0.5; G[2,1] = -0.5; G[2,2] = 0.5
    G[3,0] = 0.0; G[3,1] =  0.0; G[3,2] = 1.0

    var a_stack = InlineArray[Scalar[dtype], 8](uninitialized=True)
    var A = NDBuffer[dtype, 2,_, DimList((2, 4))](a_stack)
    A[0,0] = 1.0; A[0,1] = 1.0; A[0,2] =  1.0; A[0,3] =  0.0
    A[1,0] = 0.0; A[1,1] = 1.0; A[1,2] = -1.0; A[1,3] = -1.0
    # fmt: on

    # Temporary buffers for intermediate results
    var scratch_stack = InlineArray[Scalar[dtype], 16](uninitialized=True)
    var scratch = NDBuffer[dtype, 2, _, DimList((4, 4))](scratch_stack)
    var g_t_stack = InlineArray[Scalar[dtype], 16](uninitialized=True)
    var g_transformed = NDBuffer[dtype, 2, _, DimList((4, 4))](g_t_stack)

    # Transform kernel: G @ kernel @ G^T
    matmul[4, 3, 3, False](scratch, G, kernel)
    matmul[4, 4, 3, True](g_transformed, scratch, G)

    # Process each 2x2 output tile
    var H = signal.dim(0)
    var W = signal.dim(1)
    var Oh = H - 2
    var Ow = W - 2

    # Additional temporary buffers
    var d_stack = InlineArray[Scalar[dtype], 16](uninitialized=True)
    var d = NDBuffer[dtype, 2, _, DimList((4, 4))](d_stack)
    var m_stack = InlineArray[Scalar[dtype], 16](uninitialized=True)
    var m = NDBuffer[dtype, 2, _, DimList((4, 4))](m_stack)
    var y_stack = InlineArray[Scalar[dtype], 4](uninitialized=True)
    var y = NDBuffer[dtype, 2, _, DimList((2, 2))](y_stack)

    for i in range(0, Oh, 2):
        for j in range(0, Ow, 2):
            # Extract 4x4 input tile
            @parameter
            for di in range(4):

                @parameter
                for dj in range(4):
                    var v = Scalar[dtype](0)
                    if (i + di) < H and (j + dj) < W:
                        v = signal[i + di, j + dj]
                    d[di, dj] = v

            # Transform input: B @ d @ B^T
            matmul[4, 4, 4, False](scratch, B, d)
            matmul[4, 4, 4, True](d, scratch, B)

            # Element-wise multiplication
            for ii in range(4):
                for jj in range(4):
                    m[ii, jj] = d[ii, jj] * g_transformed[ii, jj]

            # y = A * m * A^T
            matmul[2, 4, 4, False](scratch, A, m)
            matmul[2, 2, 4, True](y, scratch, A)

            # Store results
            @parameter
            for di in range(2):

                @parameter
                for dj in range(2):
                    if i + di < Oh and j + dj < Ow:
                        output[i + di, j + dj] = y[di, dj]


fn outputs_are_close[
    dtype: DType
](
    output_naive: NDBuffer[dtype, 2, _, _],
    output_winograd: NDBuffer[dtype, 2, _, _],
    Oh: Int,
    Ow: Int,
) -> Bool:
    # Compare results
    for i in range(Oh):
        for j in range(Ow):
            if not isclose(
                output_naive[i, j],
                output_winograd[i, j],
                atol=1e-6,  # absolute error tolerance
                rtol=1e-6,  # relative error tolerance
            ):
                print("Mismatch at position (", i, ",", j, ")")
                print("Naive:", output_naive[i, j])
                print("Winograd:", output_winograd[i, j])
                print("Difference:", output_naive[i, j] - output_winograd[i, j])

                return False
    return True


# CHECK-LABEL: test_conv2d_winograd
fn test[dtype: DType, H: Int, W: Int]():  # Input Height/Width
    print("test_conv2d_winograd")
    alias Kh: Int = 3  # Filter height
    alias Kw: Int = 3  # Filter width
    alias Oh: Int = H - Kh + 1  # Output height
    alias Ow: Int = W - Kw + 1  # Output width

    # Allocate memory for input, filter, and both outputs
    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(H * W)
    var filter_ptr = UnsafePointer[Scalar[dtype]].alloc(Kh * Kw)
    var output_ptr_winograd = UnsafePointer[Scalar[dtype]].alloc(Oh * Ow)
    var output_ptr_naive = UnsafePointer[Scalar[dtype]].alloc(Oh * Ow)

    # Create NDBuffers
    var input = NDBuffer[dtype, 2, _, DimList((H, W))](input_ptr)
    var filter = NDBuffer[dtype, 2, _, DimList((Kh, Kw))](filter_ptr)
    var output_winograd = NDBuffer[dtype, 2, _, DimList((Oh, Ow))](
        output_ptr_winograd
    )
    var output_naive = NDBuffer[dtype, 2, _, DimList((Oh, Ow))](
        output_ptr_naive
    )

    # Initialize with random values
    rand[dtype](input_ptr, H * W)
    rand[dtype](filter_ptr, Kh * Kw)

    # Perform Winograd-based convolution
    winograd_2d_convolution_3x3[dtype](input, filter, output_winograd)

    # Perform Naive convolution
    alias output_shape = Index(1, 1, Oh, Ow, 1)
    alias input_shape = Index(1, 1, H, W, 1)
    alias filter_shape = Index(1, Kh, Kw, 1, 1)
    alias pad_d = Index(0, 0)
    alias pad_h = Index(0, 0)
    alias pad_w = Index(0, 0)
    alias stride = Index(1, 1, 1)
    alias dilation = Index(1, 1, 1)

    Naive2dConvolution[dtype, dtype, dtype].run(
        output_ptr_naive,
        input_ptr,
        filter_ptr,
        output_shape,
        input_shape,
        filter_shape,
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        1,
    )

    # CHECK: Succeed
    if outputs_are_close[dtype](output_naive, output_winograd, Oh, Ow):
        print("Succeed")

    # Free allocated memory
    input_ptr.free()
    filter_ptr.free()
    output_ptr_winograd.free()
    output_ptr_naive.free()

    # CHECK: Succeed
    print("Succeed")


def main():
    alias dtype = DType.float32

    # power of 2
    test[dtype, 4, 4]()
    test[dtype, 8, 8]()
    test[dtype, 16, 16]()
    test[dtype, 32, 32]()

    # Test odd sizes
    test[dtype, 3, 3]()
    test[dtype, 7, 7]()
    test[dtype, 9, 9]()

    # Non square
    test[dtype, 3, 5]()
    test[dtype, 17, 9]()
