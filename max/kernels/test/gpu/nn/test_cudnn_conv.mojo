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

from math import ceildiv, isclose
from random import rand
from sys import sizeof
from sys.info import num_physical_cores, simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    arange,
    fill,
    random,
    zero,
)
from memory import UnsafePointer
from nn.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    conv_cudnn,
    conv_gpu,
    conv_nhwc_direct,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv_utils import (
    ConvShape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)
from testing import assert_almost_equal, assert_equal

from utils.index import Index, IndexList


fn print_data[type: DType](data: UnsafePointer[Scalar[type]], dim: DimList):
    for i in range(dim.product().get()):
        print(data[i], " ", end="")
    print("")


# input: NHWC
# filer: RSCF
fn test_conv_cudnn[
    input_dim: DimList,
    filter_dim: DimList,
    output_dim: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    stride_dim: IndexList[2],
    dilation_dim: IndexList[2],
    pad_dim: IndexList[2],
](ctx: DeviceContext) raises:
    print("== test_cudnn_conv_gpu")

    var input_dim_flattened = input_dim.product().get()
    var filter_dim_flattened = filter_dim.product().get()
    var output_dim_flattened = output_dim.product().get()
    alias filter_dim_nchw = DimList(
        filter_dim.get[3](),
        filter_dim.get[2](),
        filter_dim.get[0](),
        filter_dim.get[1](),
    )

    var input_host = HostNDBuffer[input_type, 4, input_dim](input_dim)
    var filter_host = HostNDBuffer[filter_type, 4, filter_dim](filter_dim)
    var filter_nchw_host = HostNDBuffer[filter_type, 4, filter_dim_nchw](
        filter_dim_nchw
    )
    var output_ref_host = HostNDBuffer[output_type, 4, output_dim](output_dim)
    var output_host = HostNDBuffer[output_type, 4, output_dim](output_dim)

    random(input_host.tensor)
    random(filter_host.tensor)

    # Transpose filter to NCHW
    alias R = filter_dim.get[0]()
    alias S = filter_dim.get[1]()
    alias C = filter_dim.get[2]()
    alias F = filter_dim.get[3]()
    for r in range(R):
        for s in range(S):
            for c in range(C):
                for f in range(F):
                    filter_nchw_host.tensor[f, c, r, s] = filter_host.tensor[
                        r, s, c, f
                    ]

    zero(output_host.tensor)
    zero(output_ref_host.tensor)

    var input_dev = ctx.enqueue_create_buffer[input_type](input_dim_flattened)
    var filter_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var filter_nchw_dev = ctx.enqueue_create_buffer[filter_type](
        filter_dim_flattened
    )
    var output_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )
    var output_ref_dev = ctx.enqueue_create_buffer[output_type](
        output_dim_flattened
    )

    ctx.enqueue_copy(input_dev, input_host.tensor.data)
    ctx.enqueue_copy(filter_dev, filter_host.tensor.data)
    ctx.enqueue_copy(filter_nchw_dev, filter_nchw_host.tensor.data)

    conv_gpu[
        4,
        4,
        input_dim,
        filter_dim,
        output_dim,
        input_type,
        filter_type,
        output_type,
    ](
        input_dev._unsafe_ptr(),
        filter_dev._unsafe_ptr(),
        output_ref_dev._unsafe_ptr(),
        stride_dim,
        dilation_dim,
        pad_dim,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_ref_host.tensor.data, output_ref_dev)

    conv_cudnn[
        input_dim,
        DimList(
            filter_dim.get[3](),
            filter_dim.get[2](),
            filter_dim.get[0](),
            filter_dim.get[1](),
        ),
        output_dim,
        input_type,
        filter_type,
        output_type,
    ](
        input_dev._unsafe_ptr(),
        filter_nchw_dev._unsafe_ptr(),
        output_dev._unsafe_ptr(),
        stride_dim,
        dilation_dim,
        pad_dim,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_host.tensor.data, output_dev)

    # verifying results
    output_host_buf = output_host.tensor
    output_ref_host_buf = output_ref_host.tensor
    alias N = output_dim.get[0]()
    alias Hout = output_dim.get[1]()
    alias Wout = output_dim.get[2]()
    for n in range(N):
        for h in range(Hout):
            for w in range(Wout):
                for f in range(F):
                    assert_almost_equal(
                        output_host_buf[n, h, w, f],
                        output_ref_host_buf[n, h, w, f],
                        rtol=0.0001,
                    )
    print("Succeed")

    _ = input_host
    _ = filter_host
    _ = output_host
    _ = output_ref_host
    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^
    _ = output_ref_dev^


def main():
    with DeviceContext() as ctx:
        test_conv_cudnn[
            DimList(1, 64, 64, 64),  # input  (NHWC)
            DimList(
                3, 3, 64, 128
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DimList(1, 62, 62, 128),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 0),  # pad
        ](ctx)
