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

from complex import ComplexScalar
from gpu.host import DeviceContext
from gpu.host.info import is_cpu
from layout import Layout, LayoutTensor
from math import ceil
from sys.info import has_accelerator
from utils.numerics import nan

from testing import assert_almost_equal

from nn._fft.fft import (
    _cpu_fft_kernel_radix_n,
    _intra_block_fft_kernel_radix_n,
    _launch_inter_multiprocessor_fft,
)
from nn._fft.utils import _get_ordered_bases_processed_list
from ._test_values import (
    _TestValues,
    _get_test_values_2,
    _get_test_values_3,
    _get_test_values_4,
    _get_test_values_5,
    _get_test_values_6,
    _get_test_values_7,
    _get_test_values_8,
    _get_test_values_10,
    _get_test_values_16,
    _get_test_values_20,
    _get_test_values_21,
    _get_test_values_32,
    _get_test_values_35,
    _get_test_values_48,
    _get_test_values_60,
    _get_test_values_64,
    _get_test_values_100,
    _get_test_values_128,
)


fn test_fft[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    test_num: UInt,
    bases: List[UInt],
    inverse: Bool = False,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
) raises:
    constrained[len(in_layout) == 3, "in_layout must have rank 3"]()
    alias batches = in_layout.shape[0].value()
    alias sequence_length = in_layout.shape[1].value()
    alias do_rfft = in_layout.shape[2].value() == 1
    alias do_complex = in_layout.shape[2].value() == 2

    constrained[
        do_rfft or do_complex,
        "The layout should match one of: {(batches, sequence_length, 1), ",
        "(batches, sequence_length, 2)}",
    ]()
    constrained[
        out_dtype.is_floating_point(), "out_dtype must be floating point"
    ]()

    alias bases_processed = _get_ordered_bases_processed_list[
        sequence_length, bases, target
    ]()
    alias ordered_bases = bases_processed[0]
    alias processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> (UInt, List[UInt]):
        alias last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = (sequence_length // last_base) * (last_base - 1) * len(bases)
        var offsets = List[UInt](capacity=c)
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (sequence_length // base) * (base - 1)
        return val, offsets^

    alias total_offsets = _calc_total_offsets()
    alias total_twfs = total_offsets[0]
    alias twf_offsets = total_offsets[1]

    @parameter
    if is_cpu[target]():
        _cpu_fft_kernel_radix_n[
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
        ](output, x)
        return
    constrained[
        has_accelerator(), "The non-cpu implementation is for GPU only"
    ]()

    alias gpu_info = ctx.default_device_info
    alias max_threads_per_block = gpu_info.max_thread_block_size
    alias threads_per_sm = gpu_info.threads_per_sm
    alias max_threads_available = threads_per_sm * gpu_info.sm_count
    alias num_threads = sequence_length // ordered_bases[len(ordered_bases) - 1]
    alias num_blocks = UInt(
        ceil(num_threads / max_threads_per_block).cast[DType.uint]()
    )
    alias shared_mem_size = gpu_info.shared_memory_per_multiprocessor
    alias output_size = out_dtype.size_of() * sequence_length * 2
    alias twf_size = out_dtype.size_of() * total_twfs * 2

    @parameter
    if test_num == 0 or test_num == 1:
        alias batch_size = max_threads_available // num_threads
        alias func = _intra_block_fft_kernel_radix_n[
            in_dtype,
            out_dtype,
            in_layout,
            out_layout,
            length=sequence_length,
            ordered_bases=ordered_bases,
            processed_list=processed_list,
            do_rfft=do_rfft,
            inverse=inverse,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            warp_exec = (
                UInt(gpu_info.warp_size) >= batches * num_threads
            ) if test_num
            == 1 else False,
        ]

        @parameter
        for _ in range(batches // batch_size):
            ctx.enqueue_function[func](
                output, x, grid_dim=(1, batch_size), block_dim=num_threads
            )
        alias remainder = batches % batch_size

        @parameter
        if remainder > 0:
            ctx.enqueue_function[func](
                output, x, grid_dim=(1, remainder), block_dim=num_threads
            )
    elif test_num == 2 or test_num == 3:
        alias block_dim = UInt(
            ceil(num_threads / num_blocks).cast[DType.uint]()
        )
        _launch_inter_multiprocessor_fft[
            length=sequence_length,
            processed_list=processed_list,
            ordered_bases=ordered_bases,
            do_rfft=do_rfft,
            inverse=inverse,
            block_dim=block_dim,
            num_blocks=num_blocks,
            batches=batches,
            total_twfs=total_twfs,
            twf_offsets=twf_offsets,
            max_cluster_size = 0 if test_num == 2 else 8,
        ](output, x, ctx)
    else:
        # TODO: Implement for sequences > max_threads_available in the same GPU
        constrained[
            False,
            "fft for sequences longer than max_threads_available",
            "is not implemented yet. max_threads_available: ",
            String(max_threads_available),
        ]()


def test_fft_radix_n[
    dtype: DType,
    bases: List[UInt],
    test_values: _TestValues[dtype],
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
]():
    alias BATCHES = len(test_values)
    alias SIZE = len(test_values[0][0])
    alias in_dtype = dtype
    alias out_dtype = dtype
    alias in_layout = Layout.row_major(
        BATCHES, SIZE, 2
    ) if inverse else Layout.row_major(BATCHES, SIZE, 1)
    alias out_layout = Layout.row_major(BATCHES, SIZE, 2)

    @parameter
    if debug:
        print("----------------------------")
        print("Buffers for Bases:")
        var b = materialize[bases]()
        print(b.__str__().replace("UInt(", "").replace(")", ""))
        print("----------------------------")

    @parameter
    fn _eval[
        res_layout: Layout, res_origin: MutableOrigin
    ](
        result: LayoutTensor[out_dtype, res_layout, res_origin],
        scalar_in: List[Int],
        complex_out: List[ComplexScalar[out_dtype]],
    ) raises:
        @parameter
        if debug:
            print("out: ", end="")
            for i in range(SIZE):
                if i == 0:
                    print("[", result[i, 0], ", ", result[i, 1], sep="", end="")
                else:
                    print(
                        ", ", result[i, 0], ", ", result[i, 1], sep="", end=""
                    )
            print("]")
            print("expected: ", end="")

        alias ATOL = 1e-3 if dtype is DType.float64 else (
            1e-2 if dtype is DType.float32 else 1e-1
        )
        alias RTOL = 1e-5

        # gather all real parts and then the imaginary parts
        @parameter
        if inverse:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", scalar_in[i], ".0, 0.0", sep="", end="")
                    else:
                        print(", ", scalar_in[i], ".0, 0.0", sep="", end="")
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    Scalar[out_dtype](scalar_in[i]),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(result[i, 1], 0, atol=ATOL, rtol=RTOL)
        else:

            @parameter
            if debug:
                for i in range(SIZE):
                    if i == 0:
                        print("[", complex_out[i].re, ", ", sep="", end="")
                    else:
                        print(", ", complex_out[i].re, ", ", sep="", end="")
                    print(complex_out[i].im, end="")
                print("]")
            for i in range(SIZE):
                assert_almost_equal(
                    result[i, 0],
                    complex_out[i].re.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )
                assert_almost_equal(
                    result[i, 1],
                    complex_out[i].im.cast[out_dtype](),
                    atol=ATOL,
                    rtol=RTOL,
                )

    with DeviceContext() as ctx:

        @parameter
        if target == "cpu":
            var out_data = List[Scalar[in_dtype]](
                length=out_layout.size(), fill=nan[in_dtype]()
            )
            var x_data = List[Scalar[out_dtype]](
                length=in_layout.size(), fill=nan[out_dtype]()
            )
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                Span(out_data)
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                Span(x_data)
            )

            for idx, test in enumerate(materialize[test_values]()):
                alias x_layout = Layout.row_major(
                    in_layout.shape[1].value(), in_layout.shape[2].value()
                )
                var x = LayoutTensor[mut=True, in_dtype, x_layout](
                    batch_x.ptr + batch_x.stride[0]() * idx
                )
                for i in range(SIZE):

                    @parameter
                    if inverse:
                        x[i, 0] = test[1][i].re.cast[in_dtype]()
                        x[i, 1] = test[1][i].im.cast[in_dtype]()
                    else:
                        x[i, 0] = Scalar[in_dtype](test[0][i])

            test_fft[bases=bases, inverse=inverse, target=target, test_num=0](
                batch_output, batch_x, ctx
            )

            for idx, test in enumerate(materialize[test_values]()):
                alias output_layout = Layout.row_major(
                    out_layout.shape[1].value(), 2
                )
                var output = LayoutTensor[
                    mut=True, out_dtype, output_layout, batch_output.origin
                ](batch_output.ptr + batch_output.stride[0]() * idx)
                _eval(output, test[0], test[1])
        else:
            var x_data = ctx.enqueue_create_buffer[in_dtype](
                in_layout.size()
            ).enqueue_fill(nan[in_dtype]())
            var out_data = ctx.enqueue_create_buffer[out_dtype](
                out_layout.size()
            ).enqueue_fill(nan[out_dtype]())
            var batch_output = LayoutTensor[mut=True, out_dtype, out_layout](
                out_data.unsafe_ptr()
            )
            var batch_x = LayoutTensor[mut=False, in_dtype, in_layout](
                x_data.unsafe_ptr()
            )
            with x_data.map_to_host() as x_host:
                for idx, test in enumerate(materialize[test_values]()):
                    alias x_layout = Layout.row_major(
                        in_layout.shape[1].value(), in_layout.shape[2].value()
                    )
                    var x = LayoutTensor[mut=False, in_dtype, x_layout](
                        x_host.unsafe_ptr() + batch_x.stride[0]() * idx
                    )

                    for i in range(SIZE):

                        @parameter
                        if inverse:
                            x[i, 0] = test[1][i].re.cast[in_dtype]()
                            x[i, 1] = test[1][i].im.cast[in_dtype]()
                        else:
                            x[i, 0] = Scalar[in_dtype](test[0][i])

            ctx.synchronize()
            test_fft[
                bases=bases,
                inverse=inverse,
                target="gpu",
                test_num=test_num,
            ](batch_output, batch_x, ctx)
            ctx.synchronize()
            with out_data.map_to_host() as out_host:
                for idx, test in enumerate(materialize[test_values]()):
                    alias output_layout = Layout.row_major(
                        out_layout.shape[1].value(), 2
                    )
                    var output = LayoutTensor[
                        mut=True, out_dtype, output_layout, batch_output.origin
                    ](out_host.unsafe_ptr() + batch_output.stride[0]() * idx)
                    _eval(output, test[0], test[1])

        @parameter
        if debug:
            print("----------------------------")
            print("Tests passed")
            print("----------------------------")


def _test_fft[
    dtype: DType,
    func: fn[bases: List[UInt], test_values: _TestValues[dtype]] () raises,
]():
    alias L = List[UInt]

    alias values_2 = _get_test_values_2[dtype]()
    func[[2], values_2]()

    alias values_3 = _get_test_values_3[dtype]()
    func[[3], values_3]()

    alias values_4 = _get_test_values_4[dtype]()
    func[[4], values_4]()
    func[[2], values_4]()

    alias values_5 = _get_test_values_5[dtype]()
    func[[5], values_5]()

    alias values_6 = _get_test_values_6[dtype]()
    func[[6], values_6]()
    func[[3, 2], values_6]()
    func[[2, 3], values_6]()

    alias values_7 = _get_test_values_7[dtype]()
    func[[7], values_7]()

    alias values_8 = _get_test_values_8[dtype]()
    func[[8], values_8]()
    func[[2], values_8]()
    func[[4, 2], values_8]()
    func[[2, 4], values_8]()

    alias values_10 = _get_test_values_10[dtype]()
    func[[10], values_10]()
    func[[5, 2], values_10]()

    alias values_16 = _get_test_values_16[dtype]()
    func[[16], values_16]()
    func[[2], values_16]()
    func[[4], values_16]()
    func[[2, 4], values_16]()
    func[[8, 2], values_16]()
    func[[2, 8], values_16]()

    alias values_20 = _get_test_values_20[dtype]()
    func[[20], values_20]()
    func[[10, 2], values_20]()
    func[[5, 4], values_20]()
    func[[5, 2], values_20]()

    alias values_21 = _get_test_values_21[dtype]()
    func[[7, 3], values_21]()

    alias values_32 = _get_test_values_32[dtype]()
    func[[2], values_32]()
    func[[16, 2], values_32]()
    func[[8, 4], values_32]()
    func[[4, 2], values_32]()
    func[[8, 2], values_32]()

    alias values_35 = _get_test_values_35[dtype]()
    func[[7, 5], values_35]()

    alias values_48 = _get_test_values_48[dtype]()
    func[[8, 6], values_48]()
    func[[3, 2], values_48]()

    alias values_60 = _get_test_values_60[dtype]()
    func[[10, 6], values_60]()
    func[[6, 5, 2], values_60]()
    func[[5, 4, 3], values_60]()
    func[[3, 4, 5], values_60]()
    func[[5, 3, 2], values_60]()

    alias values_64 = _get_test_values_64[dtype]()
    func[[2], values_64]()
    func[[8], values_64]()
    func[[4], values_64]()
    func[[16, 4], values_64]()

    alias values_100 = _get_test_values_100[dtype]()
    func[[20, 5], values_100]()
    func[[10], values_100]()
    func[[5, 4], values_100]()

    alias values_128 = _get_test_values_128[dtype]()
    # func[[64, 2], values_128]()  # long compile times, but important to test
    # func[[32, 4], values_128]()  # long compile times, but important to test
    func[[16, 8], values_128]()
    func[[16, 4, 2], values_128]()
    func[[8, 8, 2], values_128]()
    func[[8, 4, 4], values_128]()
    func[[8, 4, 2, 2], values_128]()
    func[[8, 2, 2, 2, 2], values_128]()
    func[[4, 4, 4, 2], values_128]()
    func[[4, 4, 2, 2, 2], values_128]()
    func[[4, 2, 2, 2, 2, 2], values_128]()
    func[[2], values_128]()


alias _test[
    dtype: DType,
    inverse: Bool,
    target: StaticString,
    test_num: UInt,
    debug: Bool,
] = _test_fft[
    dtype,
    test_fft_radix_n[
        dtype, inverse=inverse, target=target, test_num=test_num, debug=debug
    ],
]


def test_fft():
    alias dtype = DType.float32
    _test[dtype, False, "cpu", 0, debug=False]()
    _test[dtype, False, "gpu", 0, debug=False]()
    _test[dtype, False, "gpu", 1, debug=False]()
    _test[dtype, False, "gpu", 2, debug=False]()
    _test[dtype, False, "gpu", 3, debug=False]()


def test_ifft():
    alias dtype = DType.float32
    _test[dtype, True, "cpu", 0, debug=False]()
    _test[dtype, True, "gpu", 0, debug=False]()
    _test[dtype, True, "gpu", 1, debug=False]()
    _test[dtype, True, "gpu", 2, debug=False]()
    _test[dtype, True, "gpu", 3, debug=False]()


def main():
    test_fft()
    test_ifft()
