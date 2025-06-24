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

from math import ceildiv
from random import rand

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from nn.conv import (
    Naive2dConvolution,
    conv3d_gpu_naive_ndhwc_qrscf,
)
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn test_conv3d_gpu[
    input_dim: DimList,
    filter_dim: DimList,
    type: DType,
    stride: IndexList[3],
    dilation: IndexList[3],
    pad: IndexList[3],
](ctx: DeviceContext) raises:
    print("test_conv3d: Testing 3D Convolution")
    alias N = input_dim.get[0]()
    alias D = input_dim.get[1]()
    alias H = input_dim.get[2]()
    alias W = input_dim.get[3]()
    alias C = input_dim.get[4]()

    alias Q = filter_dim.get[0]()
    alias R = filter_dim.get[1]()
    alias S = filter_dim.get[2]()
    alias F = filter_dim.get[4]()

    alias pad_d = IndexList[2](pad[0], pad[0])
    alias pad_h = IndexList[2](pad[1], pad[1])
    alias pad_w = IndexList[2](pad[2], pad[2])

    # compute output dimensions, just working backwards to see what the output shape will be
    alias D_out = (
        D + pad_d[0] + pad_d[1] - dilation[0] * (Q - 1) - 1
    ) // stride[0] + 1
    alias H_out = (
        H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1
    ) // stride[1] + 1
    alias W_out = (
        W + pad_w[0] + pad_w[1] - dilation[2] * (S - 1) - 1
    ) // stride[2] + 1

    alias output_dim = DimList(N, D_out, H_out, W_out, F)

    # calculate flattened sizes, gotta know how much memory we need
    var input_size = input_dim.product().get()
    var filter_size = filter_dim.product().get()
    var output_size = output_dim.product().get()

    # allocate host memory and initialize with random data
    var input_host = UnsafePointer[Scalar[type]].alloc(input_size)
    var filter_host = UnsafePointer[Scalar[type]].alloc(filter_size)
    var output_gpu_host = UnsafePointer[Scalar[type]].alloc(output_size)
    var output_ref_host = UnsafePointer[Scalar[type]].alloc(output_size)

    # initialize with random data
    rand[type](input_host, input_size)
    rand[type](filter_host, filter_size)

    # run reference implementation
    Naive2dConvolution[type, type, type].run(
        output_ref_host,
        input_host,
        filter_host,
        Index(N, D_out, H_out, W_out, F),  # output shape
        Index(N, D, H, W, C),  # input shape
        Index(Q, R, S, C, F),  # filter shape
        pad_d,
        pad_h,
        pad_w,
        (stride[0], stride[1], stride[2]),
        (dilation[0], dilation[1], dilation[2]),
        1,  # num_groups
    )
    # allocate device memory
    var input_dev = ctx.enqueue_create_buffer[type](input_size)
    var filter_dev = ctx.enqueue_create_buffer[type](filter_size)
    var output_dev = ctx.enqueue_create_buffer[type](output_size)

    # copy input and filter to device, shipping data to gpu land
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    # create ndbuffer views, making it easier to work with
    var input_buf = NDBuffer[type, 5, _, input_dim](
        input_dev._unsafe_ptr(), input_dim
    )
    var filter_buf = NDBuffer[type, 5, _, filter_dim](
        filter_dev._unsafe_ptr(), filter_dim
    )
    var output_buf = NDBuffer[type, 5, _, output_dim](
        output_dev._unsafe_ptr(), output_dim
    )

    # define grid and block dimensions for the gpu kernel
    alias block_size = 16
    var grid_dim_x = ceildiv(
        W_out * H_out, block_size
    )  # collapsed width and height into 1 dimension
    var grid_dim_y = ceildiv(D_out, block_size)  # depth is the y dimension
    var grid_dim_z = N  # batch size is the z dimension

    # run gpu implementation
    ctx.enqueue_function[
        conv3d_gpu_naive_ndhwc_qrscf[
            input_dim,
            filter_dim,
            output_dim,
            type,
            type,
            type,
            block_size,
            None,
        ]
    ](
        input_buf,
        filter_buf,
        output_buf,
        stride,
        dilation,
        pad,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_size, block_size, 1),
    )

    # copy result back to host, bringing it home
    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)

    # Verify results using assert_almost_equal
    try:
        for i in range(output_size):
            assert_almost_equal(
                output_ref_host[i], output_gpu_host[i], rtol=1e-4, atol=1e-4
            )
        print("RESULT: PASS - All elements match within tolerance")
    except:
        print("RESULT: FAIL - Elements do not match")
    finally:
        input_host.free()
        filter_host.free()
        output_gpu_host.free()
        output_ref_host.free()


fn main() raises:
    with DeviceContext() as ctx:
        # test case 1: small dimensions, starting simple
        test_conv3d_gpu[
            DimList(1, 4, 4, 4, 2),  # input (NDHWC)
            DimList(2, 2, 2, 2, 3),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # padding
        ](ctx)

        # test case 2: medium dimensions with padding
        test_conv3d_gpu[
            DimList(2, 6, 6, 6, 4),  # input (NDHWC)
            DimList(3, 3, 3, 4, 8),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 2, 2),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)

        # test case 3: non-square dimensions
        test_conv3d_gpu[
            DimList(1, 5, 7, 9, 3),  # input (NDHWC)
            DimList(2, 3, 2, 3, 4),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 1, 0),  # padding
        ](ctx)

        # test case 4: varying filter dimensions, getting creative
        test_conv3d_gpu[
            DimList(1, 9, 8, 5, 1),  # input (NDHWC)
            DimList(2, 2, 3, 1, 32),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 3, 2),  # stride - mixed stride values
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](0, 0, 0),  # padding
        ](ctx)

        # test case 5: with padding on all dimensions
        test_conv3d_gpu[
            DimList(1, 5, 7, 6, 7),  # input (NDHWC)
            DimList(3, 4, 3, 7, 24),  # filter (QRSCF)
            DType.float32,
            IndexList[3](1, 1, 1),  # stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)

        # test case 6: large dimensions with asymmetric padding
        test_conv3d_gpu[
            DimList(1, 10, 11, 6, 2),  # input (NDHWC)
            DimList(3, 4, 3, 2, 31),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 3, 1),  # stride - mixed stride
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 2, 1),  # padding - asymmetric
        ](ctx)

        # test case 7: 3d-unet style small dimensions
        test_conv3d_gpu[
            DimList(1, 8, 8, 8, 320),  # input (NDHWC)
            DimList(3, 3, 3, 320, 320),  # filter (QRSCF)
            DType.float32,
            IndexList[3](2, 2, 2),  # stride - downsampling
            IndexList[3](1, 1, 1),  # dilation
            IndexList[3](1, 1, 1),  # padding
        ](ctx)
