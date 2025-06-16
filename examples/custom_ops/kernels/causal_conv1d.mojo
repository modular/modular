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

"""
The code implements the causal 1D convolution based
similar to the code in
https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu
And intended to showcase use of LayoutTensor use example of implementing an
Improved performance compared to naive implementation
"""


from algorithm import parallelize_over_rows
from compiler import register
from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_idx
from gpu.memory import AddressSpace
from gpu.sync import barrier
from layout import Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout.math import max, sum
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from math import exp, ceildiv

from utils import Index
from utils.index import IndexList
from python import Python, PythonObject
from os import abort
from sys import argv

from memory import UnsafePointer


@register("causal_conv1d_cpu")
struct CausalConv1Dcpu:
    """Registers the `causal_conv1d_cpu` op, allowing python to use it from the `max`
    package.
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        //,  # Forces the previous two params to be inferred from the args
        threads: Int,  # Number of threads to in a grid processing an input batch
        elements: Int,  # Number of elements to process in a single thread
        width: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        output2d: OutputTensor[dtype=dtype, rank=2],
        x: InputTensor[dtype=dtype, rank=3],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # print("Running on CPU")
        causal_conv1d_cpu[dtype, threads, elements, width](
            x, weight, bias, output
        )


@register("causal_conv1d_v1")
struct CausalConv1Dgpu:
    """Registers the `causal_conv1d_v1` op that runs ont he gpu.
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        //,  # Forces the previous two params to be inferred from the args
        threads: Int,  # Number of threads to in a grid processing an input batch
        elements: Int,  # Number of elements to process in a single thread
        width: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        output2d: OutputTensor[dtype=dtype, rank=2],
        x: InputTensor[dtype=dtype, rank=3],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        dev_ctx = ctx.get_device_context()
        # print("Running on GPU")
        x_l = x.to_layout_tensor()
        w_l = weight.to_layout_tensor()
        b_l = bias.to_layout_tensor()
        o_l = output.to_layout_tensor()
        xx_2d = output2d.to_layout_tensor()
        # print ("Running on GPU v1 kernel")
        causal_conv1d_gpu[dtype, threads, elements, width](
            dev_ctx, x_l, w_l, b_l, o_l, xx_2d
        )


# The conv1D CPU reference code to test for functional correctness
@always_inline
fn causal_conv1d_cpu[
    dtype: DType, threads: Int, elements: Int, width: Int
](
    input: InputTensor[dtype=dtype, rank=3],
    weight: InputTensor[dtype=dtype, rank=2],
    bias: InputTensor[dtype=dtype, rank=1],
    output: OutputTensor[dtype=dtype, rank=3],
):
    alias kChunkSize = threads * elements
    var batch = input.shape()[0]
    var seq_length = input.shape()[2]
    var channels = input.shape()[1]
    var x_vals: SIMD[dtype, elements * 2]
    var prev_input_chunk: SIMD[dtype, elements]
    var out_vals: SIMD[dtype, elements] = 0
    var W: SIMD[dtype, width]

    n_chunks = seq_length // kChunkSize + 1

    for batch_id in range(batch):
        for channel_id in range(channels):
            W = weight.load[width, 2](Index(channel_id, 0))
            B = bias.load[1, 1](channel_id)
            prev_input_chunk = 0
            for chunk in range(n_chunks):
                var tmp: SIMD[dtype, width]
                input_chunk = input.load[elements, 3](
                    Index(batch_id, channel_id, chunk * kChunkSize)
                )

                x_vals = prev_input_chunk.join(input_chunk)
                prev_input_chunk = input_chunk

                @parameter
                for i in range(elements):
                    tmp = (
                        W
                        * x_vals.slice[
                            width, offset = elements + i - (width - 1)
                        ]()
                    )
                    tmp2 = B[0] + tmp.reduce_add[1]()
                    out_vals[i] = tmp2 / (1 + exp(-tmp2))
                # print("writing:", out_vals, " To:", batch_id, channel_id, (chunk + i)*elements)
                output.store[elements, 3](
                    Index(batch_id, channel_id, chunk * elements), out_vals
                )


# The conv1D gpu code is an example implementation that uses LayoutTensor
# to perform a 1D filter on the signal dimension using SIMD access pattern
# and SIMD operations on the GPU, this code wasn't not intended to achieve
# the most optimal implementation, but to show a meaningful gains versus a
# naive implementation that works well across GPU architectures and using
# float32 and bfloat16
fn causal_conv1d_kernel[
    dtype: DType,
    i_layout: Layout,
    w_layout: Layout,
    b_layout: Layout,
    layout_2d: Layout,
    threads: Int,
    elements: Int,
    width: Int,
](
    input: LayoutTensor[dtype, i_layout, MutableAnyOrigin],
    weight: LayoutTensor[dtype, w_layout, MutableAnyOrigin],
    bias: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    output: LayoutTensor[dtype, i_layout, MutableAnyOrigin],
):
    var seq_length = input.shape[2]()

    # Use the 3D grid to iterate over the batch, channel and sequence length dimensions
    tidx = thread_idx.x
    batch_id = block_idx.z
    channel_id = block_idx.y
    chunk_id = block_idx.x
    kChunkSize = block_dim.x

    # Use of SIMD types of efficient load and compute processing
    # Using SIMD mojo primitives that give a reasonable performance
    # when moving between different architectures and better than
    # The naive implementation
    var x_vals: SIMD[dtype, elements * 2]
    var out_vals: SIMD[dtype, elements] = 0
    var tmp: SIMD[dtype, width]
    var B: SIMD[dtype, 1]
    var W: SIMD[dtype, width]
    var prev_input_chunk: SIMD[dtype, elements]
    var input_chunk: SIMD[dtype, elements]

    W_v = weight.vectorize[1, width]()
    W = rebind[__type_of(W)](W_v[channel_id])
    B = rebind[__type_of(B)](bias[channel_id])

    var input_v = input.reshape[layout_2d]().vectorize[1, elements]()
    var output_v = output.reshape[layout_2d]().vectorize[1, elements]()

    nChannels = input.shape[1]()
    n_chunks = seq_length // kChunkSize + 1

    if (tidx > 0) or (chunk_id > 0):
        prev_input_chunk = rebind[__type_of(prev_input_chunk)](
            input_v[
                batch_id * nChannels + channel_id,
                (chunk_id * kChunkSize + tidx - 1),
            ]
        )
    else:
        prev_input_chunk = 0

    input_chunk = rebind[__type_of(input_chunk)](
        input_v[
            batch_id * nChannels + channel_id, (chunk_id * kChunkSize + tidx)
        ]
    )

    x_vals = prev_input_chunk.join(input_chunk)

    # The convolution filter operates on the preceding input signal positions
    # Use SIMD primitives to implement the dot product
    @parameter
    for i in range(elements):
        tmp = W * x_vals.slice[width, offset = elements + i - (width - 1)]()
        tmp2 = B[0] + tmp.reduce_add[1]()
        out_vals[i] = tmp2 / (1 + exp(-tmp2))

    output_v[
        batch_id * nChannels + channel_id, tidx + chunk_id * kChunkSize
    ] = rebind[__type_of(output_v[0, 0])](out_vals)


# Mojo operation using LayoutTensor launching gpu kernel
# with LayoutTensor
def causal_conv1d_gpu[
    dtype: DType,
    threads: Int,
    elements: Int,
    width: Int,
](
    ctx: DeviceContext,
    input: LayoutTensor,
    weight: LayoutTensor,
    bias: LayoutTensor,
    output: LayoutTensor,
    xx2D: LayoutTensor,
):
    alias kernel_func = causal_conv1d_kernel[
        dtype,
        input.layout,
        weight.layout,
        bias.layout,
        xx2D.layout,
        threads,
        elements,
        width,
    ]

    # Map the problem to the 3D grid to iterate over
    # the batch, channel and chunk the sequence length
    ctx.enqueue_function[kernel_func, dump_asm=False](
        input,
        weight,
        bias,
        output,
        grid_dim=(
            ceildiv(input.shape[2](), threads * elements),
            input.shape[1](),
            input.shape[0](),
        ),
        block_dim=(threads),
    )
