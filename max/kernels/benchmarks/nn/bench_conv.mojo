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

from math import align_up, ceildiv
from random import rand
from sys import simdwidthof, sizeof
from sys.param_env import env_get_int, env_get_string

from benchmark import *
from benchmark import keep
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.conv import ConvDirectNHWC, ConvInfoStatic
from nn.conv_utils import (
    ConvShape,
    extend_shape,
    get_direct_conv_micro_kernel_width,
)

from utils import IndexList
from utils.index import Index


fn bench_conv(mut m: Bench, spec: ConvSpec) raises:
    alias input_type = spec.static_info.input_type
    alias filter_type = spec.static_info.filter_type
    alias output_type = spec.static_info.output_type

    # Alignment in terms of number of elmements.
    alias alignment = 64
    alias input_align = alignment // sizeof[input_type]()
    alias filter_align = alignment // sizeof[filter_type]()
    alias output_align = alignment // sizeof[output_type]()

    alias simd_size = simdwidthof[filter_type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    var f_per_group = spec.f // spec.num_groups

    var output_dims = IndexList[spec.static_info.rank](1)

    @parameter
    for i in range(spec.static_info.rank):
        output_dims[i] = (
            spec.input_dims[i]
            + spec.pad[2 * i]
            + spec.pad[2 * i + 1]
            - spec.dilation[i] * (spec.filter_dims[i] - 1)
            - 1
        ) // spec.stride[i] + 1

    var packed_filter_shape = IndexList[spec.static_info.rank + 3](1)

    @parameter
    for i in range(spec.static_info.rank):
        packed_filter_shape[i + 1] = output_dims[i]
    packed_filter_shape[0] = spec.num_groups * ceildiv(
        f_per_group, micro_kernel_f_size
    )
    packed_filter_shape[spec.static_info.rank + 1] = spec.c
    packed_filter_shape[spec.static_info.rank + 2] = micro_kernel_f_size

    # Input and output shape, sizes
    var input_shape = extend_shape(spec.input_dims, spec.n, spec.c)
    var input_alloc_size = align_up(input_shape.flattened_length(), input_align)
    var filter_alloc_size = align_up(
        packed_filter_shape.flattened_length(), filter_align
    )
    var output_shape = extend_shape(output_dims, spec.n, spec.f)
    var output_alloc_size = align_up(
        output_shape.flattened_length(), output_align
    )

    # Set the total buffer allocation to be 4x L3 cache.
    alias MB = 1024 * 1024
    alias L3_cache = env_get_int["L3SIZE", 24]() * MB
    var size_per_copy = (
        input_alloc_size * sizeof[input_type]()
        + filter_alloc_size * sizeof[filter_type]()
    )
    var num_copies = ceildiv(4 * L3_cache, size_per_copy)

    # Allocate input and output buffers.
    var input_ptr = UnsafePointer[
        Scalar[input_type], alignment=alignment
    ].alloc(input_alloc_size * num_copies)
    var filter_ptr = UnsafePointer[
        Scalar[filter_type], alignment=alignment
    ].alloc(num_copies * filter_alloc_size)
    var output_ptr = UnsafePointer[
        Scalar[output_type], alignment=alignment
    ].alloc(num_copies * output_alloc_size)

    rand[input_type](input_ptr, num_copies * input_alloc_size)
    rand[filter_type](filter_ptr, num_copies * filter_alloc_size)

    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

    @parameter
    if spec.static_info.rank == 1:
        pad_w = Index(spec.pad[0], spec.pad[1])
    elif spec.static_info.rank == 2:
        pad_h = Index(spec.pad[0], spec.pad[1])
        pad_w = Index(spec.pad[2], spec.pad[3])
    elif spec.static_info.rank == 3:
        pad_d = Index(spec.pad[0], spec.pad[1])
        pad_h = Index(spec.pad[2], spec.pad[3])
        pad_w = Index(spec.pad[4], spec.pad[5])

    var conv_shape = ConvShape[spec.static_info.rank](
        n=spec.n,
        input_dims=spec.input_dims,
        output_dims=output_dims,
        filter_dims=spec.filter_dims,
        c=spec.c,
        f=spec.f,
        stride=spec.stride,
        dilation=spec.dilation,
        pad_d=pad_d,
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=spec.num_groups,
    )

    @parameter
    @always_inline
    fn bench_conv_wrapper(
        mut b: Bencher, concrete_spec: ConvSpec[spec.static_info]
    ) raises:
        # Count the iteration to decide which input copy to use.
        var counter = 0

        @always_inline
        @parameter
        fn bench_fn():
            var input = NDBuffer[input_type, spec.static_info.rank + 2](
                input_ptr + (counter % num_copies) * input_alloc_size,
                input_shape,
            )
            var filter = NDBuffer[filter_type, spec.static_info.rank + 3](
                filter_ptr + (counter % num_copies) * filter_alloc_size,
                packed_filter_shape,
            )
            var output = NDBuffer[output_type, spec.static_info.rank + 2](
                output_ptr + (counter % num_copies) * output_alloc_size,
                output_shape,
            )

            try:
                ConvDirectNHWC[
                    spec.static_info.rank + 2,
                    spec.static_info.rank + 3,
                    spec.static_info.rank + 2,
                    _,
                    _,
                    _,
                    DimList.create_unknown[spec.static_info.rank + 2](),
                    DimList.create_unknown[spec.static_info.rank + 3](),
                    DimList.create_unknown[spec.static_info.rank + 2](),
                    input_type,
                    filter_type,
                    output_type,
                    True,
                    ConvInfoStatic[spec.static_info.rank + 2 - 2](),
                ].run(
                    output,
                    input,
                    filter,
                    rebind[ConvShape[spec.static_info.rank + 2 - 2]](
                        conv_shape
                    ),
                )

                counter += 1

            except e:
                print(e)

            keep(output.data)

        b.iter[bench_fn]()

    m.bench_with_input[ConvSpec[spec.static_info], bench_conv_wrapper](
        BenchId("Conv", String(spec)),
        spec,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(BenchMetric.elements, spec.flops()),
    )

    input_ptr.free()
    filter_ptr.free()
    output_ptr.free()


@fieldwise_init
struct ConvSpecStatic(Copyable, Movable):
    # Conv rank, 1d, 2d, or 3d. The input rank is rank + 2.
    var rank: Int
    var input_type: DType
    var filter_type: DType
    var output_type: DType


@fieldwise_init
struct ConvSpec[static_info: ConvSpecStatic](Stringable, Copyable, Movable):
    var n: Int
    var input_dims: IndexList[static_info.rank]
    var c: Int
    var filter_dims: IndexList[static_info.rank]
    var f: Int
    var stride: IndexList[static_info.rank]
    var dilation: IndexList[static_info.rank]
    var pad: IndexList[2 * static_info.rank]
    var num_groups: Int

    @no_inline
    fn __str__(self) -> String:
        # fmt: off
        return String(
            "n=", self.n,
            ";input=", self.input_dims,
            ";c=", self.c,
            ";f=", self.f,
            ";filter=", self.filter_dims,
            ";stride=", self.stride,
            ";padding=", self.pad,
        )
        # fmt: on

    fn flops(self) -> Int:
        var output_dims = IndexList[static_info.rank](1)

        @parameter
        for i in range(static_info.rank):
            output_dims[i] = (
                self.input_dims[i]
                + self.pad[2 * i]
                + self.pad[2 * i + 1]
                - self.dilation[i] * (self.filter_dims[i] - 1)
                - 1
            ) // self.stride[i] + 1

        return (
            2
            * self.n
            * output_dims.flattened_length()
            * self.filter_dims.flattened_length()
            * self.c
            * self.f
        )


def main():
    var m = Bench(BenchConfig())

    alias fp32_1d = ConvSpecStatic(
        rank=1,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    fn rebind1d(idx: IndexList[1]) -> IndexList[fp32_1d.rank]:
        return rebind[IndexList[fp32_1d.rank]](idx)

    @always_inline
    fn rebind1d_pad(idx: IndexList[2]) -> IndexList[2 * fp32_1d.rank]:
        return rebind[IndexList[2 * fp32_1d.rank]](idx)

    # fmt: off
    @always_inline
    fn spec1d(N: Int, W: Int, C: Int, S: Int, F: Int, st: Int, di: Int, \
        pa: IndexList[2], ng: Int
    ) -> ConvSpec[fp32_1d]:
        return (
            ConvSpec[fp32_1d](
                n=N,
                input_dims=rebind1d(Index(W)),
                c=C,
                filter_dims=rebind1d(Index(S)),
                f=F,
                stride=rebind1d(Index(st)),
                dilation=rebind1d(Index(di)),
                pad=rebind1d_pad(pa),
                num_groups=ng,
            )
        )
    # fmt: on

    alias fp32_2d = ConvSpecStatic(
        rank=2,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    fn rebind2d(idx: IndexList[2]) -> IndexList[fp32_2d.rank]:
        return rebind[IndexList[fp32_2d.rank]](idx)

    @always_inline
    fn rebind2d_pad(idx: IndexList[4]) -> IndexList[2 * fp32_2d.rank]:
        return rebind[IndexList[2 * fp32_2d.rank]](idx)

    @always_inline
    fn spec2d(
        N: Int,
        H: Int,
        W: Int,
        C: Int,
        R: Int,
        S: Int,
        F: Int,
        st: IndexList[2],
        di: IndexList[2],
        pa: IndexList[4],
        ng: Int,
    ) -> ConvSpec[fp32_2d]:
        return ConvSpec[fp32_2d](
            n=N,
            input_dims=rebind2d(Index(H, W)),
            c=C,
            filter_dims=rebind2d(Index(R, S)),
            f=F,
            stride=rebind2d(st),
            dilation=rebind2d(di),
            pad=rebind2d_pad(pa),
            num_groups=ng,
        )

    alias fp32_3d = ConvSpecStatic(
        rank=3,
        input_type=DType.float32,
        filter_type=DType.float32,
        output_type=DType.float32,
    )

    @always_inline
    fn rebind3d(idx: IndexList[3]) -> IndexList[fp32_3d.rank]:
        return rebind[IndexList[fp32_3d.rank]](idx)

    @always_inline
    fn rebind3d_pad(idx: IndexList[6]) -> IndexList[3 * fp32_3d.rank]:
        return rebind[IndexList[3 * fp32_3d.rank]](idx)

    # 1D benchmarks for wavlm
    @parameter
    if env_get_string["model", "walvm"]() == "wavlm":
        bench_conv(m, spec1d(2, 16000, 1, 10, 512, 5, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 3199, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 1599, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 799, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 399, 512, 3, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 199, 512, 2, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 99, 512, 2, 512, 2, 1, Index(0, 0), 1))
        bench_conv(m, spec1d(2, 49, 1024, 128, 1024, 1, 1, Index(64, 64), 16))
    # fmt: off
    # 2D benchmarks for resnet
    elif env_get_string["model", "wavlm"]() == "resnet50":
        bench_conv(m, spec2d(1, 14, 14, 256, 3, 3, 256, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 56, 56,  64, 3, 3,  64, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 28, 28, 128, 3, 3, 128, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 7, 7,   512, 3, 3, 512, Index(1, 1), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 224, 224, 3, 7, 7,  64, Index(2, 2), Index(1, 1), Index(3, 3, 3, 3), 1))
        bench_conv(m, spec2d(1, 56, 56, 128, 3, 3, 128, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 28, 28, 256, 3, 3, 256, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 14, 14, 512, 3, 3, 512, Index(2, 2), Index(1, 1), Index(1, 1, 1, 1), 1))
        bench_conv(m, spec2d(1, 56, 56,  256, 1, 1, 512, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
        bench_conv(m, spec2d(1, 28, 28,  512, 1, 1, 1024, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
        bench_conv(m, spec2d(1, 14, 14, 1024, 1, 1, 2048, Index(2, 2), Index(1, 1), Index(0, 0, 0, 0), 1))
    # fmt: on

    m.dump_report()
