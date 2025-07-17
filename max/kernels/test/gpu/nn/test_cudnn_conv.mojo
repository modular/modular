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


from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
    zero,
)
from nn.conv import (
    conv_cudnn,
    conv_gpu,
)
from testing import assert_almost_equal

from utils.index import IndexList


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
    num_groups: Int = 1,
](ctx: DeviceContext) raises:
    print(
        "== test_cudnn_conv_gpu: dtype_in=",
        input_type,
        " dtype_filter=",
        filter_type,
        " dtype_out=",
        output_type,
        " num_groups=",
        num_groups,
    )

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

    var input_dev = DeviceNDBuffer[input_type, 4, input_dim](input_dim, ctx=ctx)
    var filter_dev = DeviceNDBuffer[filter_type, 4, filter_dim](
        filter_dim, ctx=ctx
    )
    var filter_nchw_dev = DeviceNDBuffer[filter_type, 4, filter_dim_nchw](
        filter_dim_nchw, ctx=ctx
    )
    var output_dev = DeviceNDBuffer[output_type, 4, output_dim](
        output_dim, ctx=ctx
    )
    var output_ref_dev = DeviceNDBuffer[output_type, 4, output_dim](
        output_dim, ctx=ctx
    )

    ctx.enqueue_copy(input_dev.buffer, input_host.tensor.data)
    ctx.enqueue_copy(filter_dev.buffer, filter_host.tensor.data)
    ctx.enqueue_copy(filter_nchw_dev.buffer, filter_nchw_host.tensor.data)

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
        input_dev.tensor,
        filter_dev.tensor,
        output_ref_dev.tensor,
        stride_dim,
        dilation_dim,
        pad_dim,
        num_groups,
        ctx,
    )

    conv_cudnn[input_type, filter_type, output_type](
        input_dev.tensor,
        filter_nchw_dev.tensor,
        output_dev.tensor,
        stride_dim,
        dilation_dim,
        pad_dim,
        num_groups,
        ctx,
    )

    ctx.enqueue_copy(output_ref_host.tensor.data, output_ref_dev.buffer)
    ctx.enqueue_copy(output_host.tensor.data, output_dev.buffer)

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
                        rtol=0.01,
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
        # Test configurations for data types.
        alias dtype_configs = (DType.float32, DType.float16, DType.bfloat16)

        test_conv_cudnn[
            DimList(1, 1, 550, 1024),  # input  (NHWC)
            DimList(
                1, 7, 1024, 1024
            ),  # filter (RSCF) (height, width, in_channels, out_channels)
            DimList(1, 1, 550, 1024),  # output (NHWC)
            DType.float32,
            DType.float32,
            DType.float32,
            IndexList[2](1, 1),  # stride
            IndexList[2](1, 1),  # dilation
            IndexList[2](0, 3),  # pad
        ](ctx)

        # Test different data types.
        @parameter
        for i in range(len(dtype_configs)):
            alias dtype = dtype_configs[i]

            test_conv_cudnn[
                DimList(1, 8, 8, 16),  # input  (NHWC)
                DimList(3, 3, 16, 32),  # filter (RSCF)
                DimList(1, 6, 6, 32),  # output (NHWC)
                dtype,
                dtype,
                dtype,
                IndexList[2](1, 1),  # stride
                IndexList[2](1, 1),  # dilation
                IndexList[2](0, 0),  # pad
            ](ctx)

        # Test grouped convolutions
        # NOTE: Grouped convolutions are not supported by conv_gpu's naive implementation,
        # so we cannot validate against it. These tests would need a different reference
        # implementation or manual validation.

        # TODO(KERN-1846): Add grouped convolution tests once we have a proper
        # reference implementation.
        # The following configurations would be tested:
        # - Standard conv with num_groups=1
        # - 2 groups
        # - 4 groups
        # - Depthwise convolution (num_groups = in_channels)
        # - Grouped convolution with float16
