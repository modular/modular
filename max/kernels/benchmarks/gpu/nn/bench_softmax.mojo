# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.random import random_float64
from std.sys import (
    is_defined,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    simd_width_of,
)

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from std.runtime.asyncrt import DeviceContextPtr
from internal_utils import (
    arg_parse,
    get_defined_shape,
    int_list_to_tuple,
    update_bench_config_args,
)
from layout import Coord, TileTensor, row_major
from nn.softmax import (
    softmax_benchmark_candidate,
    softmax_benchmark_current,
    softmax_with_temperature,
)

from std.utils.index import IndexList


@always_inline
def launch_softmax_current[
    dtype: DType, rank: Int
](
    data_buf: TileTensor[dtype, ...],
    out_buf: TileTensor[mut=True, dtype, ...],
    context: DeviceContextPtr,
) raises:
    softmax_benchmark_current[dtype, simd_width_of[dtype](), rank](
        data_buf, out_buf, rank - 1, context
    )


@always_inline
def benchmark_candidate_uses_variant_one[
    dtype: DType, rank: Int, shape: IndexList[rank]
]() -> Bool:
    comptime if dtype == DType.bfloat16 and rank == 2:
        return shape[1] == 128256 and (shape[0] == 32 or shape[0] == 128)
    return False


@always_inline
def launch_softmax_candidate[
    dtype: DType, rank: Int, shape: IndexList[rank]
](
    data_buf: TileTensor[dtype, ...],
    out_buf: TileTensor[mut=True, dtype, ...],
    context: DeviceContextPtr,
) raises:
    comptime if benchmark_candidate_uses_variant_one[dtype, rank, shape]():
        softmax_benchmark_candidate[
            dtype, simd_width_of[dtype](), rank
        ](
            data_buf, out_buf, rank - 1, context
        )
    else:
        softmax_benchmark_current[dtype, simd_width_of[dtype](), rank](
            data_buf, out_buf, rank - 1, context
        )


@always_inline
def _assert_bench_impl_mode[bench_impl_mode: Int]():
    comptime assert 0 <= bench_impl_mode and bench_impl_mode <= 2


@always_inline
def bench_impl_name[bench_impl_mode: Int]() -> String:
    _assert_bench_impl_mode[bench_impl_mode]()
    comptime if bench_impl_mode == 1:
        return String("current_only")
    elif bench_impl_mode == 2:
        return String("candidate_only")
    return String("mixed")


@always_inline
def decorate_input_suffix[
    bench_impl_mode: Int
](input_suffix: String) -> String:
    return String(
        "bench_impl=", bench_impl_name[bench_impl_mode](), "/", input_suffix
    )


@always_inline
def replay_slot_name(slot_order_mode: Int, slot_position: Int) -> String:
    if slot_order_mode == 1:
        if slot_position == 0:
            return String("candidate_slot")
        if slot_position == 1:
            return String("current_post")
        return String("current_pre")
    if slot_position == 0:
        return String("current_pre")
    if slot_position == 1:
        return String("candidate_slot")
    return String("current_post")


@always_inline
def replay_slot_uses_candidate(slot_name: String) -> Bool:
    return slot_name == "candidate_slot"


@always_inline
def slot_order_mode_name(slot_order_mode: Int) -> String:
    if slot_order_mode == 1:
        return String("candidate_first")
    return String("default")


@always_inline
def launch_bench_softmax_variant[
    bench_impl_mode: Int,
    use_candidate: Bool,
    dtype: DType,
    rank: Int,
    shape: IndexList[rank],
](
    data_buf: TileTensor[dtype, ...],
    out_buf: TileTensor[mut=True, dtype, ...],
    context: DeviceContextPtr,
) raises:
    _assert_bench_impl_mode[bench_impl_mode]()
    comptime if bench_impl_mode == 2:
        # Keep unchanged control shapes on the exact current wrapper so
        # current-only and candidate-only split binaries stay
        # codegen-identical outside the active benchmark-only probe.
        comptime if benchmark_candidate_uses_variant_one[dtype, rank, shape]():
            launch_softmax_candidate[dtype, rank, shape](
                data_buf, out_buf, context
            )
        else:
            launch_softmax_current[dtype, rank](data_buf, out_buf, context)
    elif bench_impl_mode == 1:
        launch_softmax_current[dtype, rank](data_buf, out_buf, context)
    else:
        # In mixed mode, keep unchanged controls on the exact current wrapper
        # for both slot types so a same-.so current/candidate/current triplet
        # only differs on the active benchmark-only candidate path.
        comptime if use_candidate and benchmark_candidate_uses_variant_one[
            dtype, rank, shape
        ]():
            launch_softmax_candidate[dtype, rank, shape](
                data_buf, out_buf, context
            )
        else:
            launch_softmax_current[dtype, rank](data_buf, out_buf, context)


def register_softmax_variant[
    rank: Int,
    //,
    bench_impl_mode: Int,
    use_candidate: Bool,
    dtype: DType,
    shape: IndexList[rank],
](
    mut b: Bench,
    fn_name: String,
    data_buf: TileTensor[dtype, ...],
    out_buf: TileTensor[mut=True, dtype, ...],
    context: DeviceContextPtr,
    input_suffix: String,
) raises:
    var bench_input_id = String(
        fn_name, "/", dtype, "/", shape, "/", input_suffix
    )

    @always_inline
    @__copy_capture(data_buf, out_buf, context)
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            _ = ctx
            launch_bench_softmax_variant[
                bench_impl_mode, use_candidate, dtype, rank, shape
            ](data_buf, out_buf, context)

        b.iter_custom[kernel_launch](context.get_device_context())

    b.bench_function[bench_fn](BenchId("softmax", input_id=bench_input_id))


@always_inline
def should_register_replay_slot(
    replay_pass: Int,
    replay_pass_filter: Int,
    slot_name: String,
    slot_filter: String,
) -> Bool:
    if replay_pass_filter > 0 and replay_pass != replay_pass_filter:
        return False
    if slot_filter != "" and slot_filter != slot_name:
        return False
    return True


def launch_softmax_variant_once[
    rank: Int,
    //,
    bench_impl_mode: Int,
    dtype: DType,
    shape: IndexList[rank],
](
    ctx: DeviceContext,
    variant: String = "current",
) raises:
    _assert_bench_impl_mode[bench_impl_mode]()
    comptime assert (
        bench_impl_mode != 0
    ), "profile-once only supports current-only or candidate-only binaries"
    _ = variant
    comptime total = shape.flattened_length()

    var data_h = alloc[Scalar[dtype]](total)
    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var context = DeviceContextPtr(ctx)

    ctx.enqueue_copy(data_d, data_h)
    # Nsight single-launch debugging should only observe the kernel train for
    # the requested softmax path, not the staging memcpy.
    ctx.synchronize()

    comptime if bench_impl_mode == 2:
        launch_bench_softmax_variant[
            bench_impl_mode, True, dtype, rank, shape
        ](data_buf, out_buf, context)
    else:
        launch_bench_softmax_variant[
            bench_impl_mode, False, dtype, rank, shape
        ](data_buf, out_buf, context)
    ctx.synchronize()

    _ = data_d
    _ = out_d
    data_h.free()


def bench_softmax_gpu[
    rank: Int, //, bench_impl_mode: Int, dtype: DType, shape: IndexList[rank]
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    variant: String = "current",
    replay_passes: Int = 1,
    slot_filter: String = "",
    replay_pass_filter: Int = 0,
    slot_order_mode: Int = 0,
) raises:
    _assert_bench_impl_mode[bench_impl_mode]()
    if slot_order_mode < 0 or slot_order_mode > 1:
        raise Error("slot-order-mode must be 0 or 1")
    comptime cols = shape[rank - 1]
    comptime total = shape.flattened_length()

    var data_h = alloc[Scalar[dtype]](total)

    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var context = DeviceContextPtr(ctx)

    ctx.enqueue_copy(data_d, data_h)

    if replay_passes > 1:
        for replay_pass in range(1, replay_passes + 1):
            for slot_position in range(3):
                var slot_name = replay_slot_name(
                    slot_order_mode, slot_position
                )
                if should_register_replay_slot(
                    replay_pass,
                    replay_pass_filter,
                    slot_name,
                    slot_filter,
                ):
                    if replay_slot_uses_candidate(slot_name):
                        register_softmax_variant[
                            bench_impl_mode, True, dtype, shape
                        ](
                            b,
                            fn_name,
                            data_buf,
                            out_buf,
                            context,
                            decorate_input_suffix[bench_impl_mode](String(
                                "slot_order_mode=",
                                slot_order_mode_name(slot_order_mode),
                                "/slot_position=",
                                slot_position + 1,
                                "/replay_pass=",
                                replay_pass,
                                "/slot=",
                                slot_name,
                            )),
                        )
                    else:
                        register_softmax_variant[
                            bench_impl_mode, False, dtype, shape
                        ](
                            b,
                            fn_name,
                            data_buf,
                            out_buf,
                            context,
                            decorate_input_suffix[bench_impl_mode](String(
                                "slot_order_mode=",
                                slot_order_mode_name(slot_order_mode),
                                "/slot_position=",
                                slot_position + 1,
                                "/replay_pass=",
                                replay_pass,
                                "/slot=",
                                slot_name,
                            )),
                        )
    else:
        if variant.startswith("candidate"):
            register_softmax_variant[
                bench_impl_mode, True, dtype, shape
            ](
                b,
                fn_name,
                data_buf,
                out_buf,
                context,
                decorate_input_suffix[bench_impl_mode](String(
                    "variant=", variant
                )),
            )
        else:
            register_softmax_variant[
                bench_impl_mode, False, dtype, shape
            ](
                b,
                fn_name,
                data_buf,
                out_buf,
                context,
                decorate_input_suffix[bench_impl_mode](String(
                    "variant=", variant
                )),
            )

    ctx.synchronize()

    _ = data_d
    _ = out_d

    data_h.free()


def bench_softmax_with_temperature_gpu[
    dtype: DType, shape: IndexList
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    temperature: Float32,
) raises:
    comptime total = shape.flattened_length()
    comptime rows = shape[0]

    var data_h = alloc[Scalar[dtype]](total)
    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    ctx.enqueue_copy(data_d, data_h)

    var temp = temperature

    @always_inline
    @__copy_capture(data_buf, out_buf, temp)
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            softmax_with_temperature(ctx, data_buf, out_buf, temp)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "softmax_with_temperature",
            input_id=String(fn_name, "/", dtype, "/", shape, "/T=", temp),
        )
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d

    data_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "256x256"]()
    ]()
    # Benchmark-only no-op define that lets kbench force a new shared-library
    # build key without changing the measured kernel path.
    comptime build_nonce = get_defined_int["build_nonce", 0]()
    _ = build_nonce
    comptime bench_impl_mode = get_defined_int["bench_impl_mode", 0]()
    var variant = arg_parse("variant", "current")
    var replay_passes = arg_parse("replay-passes", 1)
    var slot_filter = arg_parse("slot-filter", "")
    var replay_pass_filter = arg_parse("replay-pass-filter", 0)
    var slot_order_mode = arg_parse("slot-order-mode", 0)
    var profile_once = arg_parse("profile-once", 0)
    var m = Bench(BenchConfig(num_repetitions=1))
    update_bench_config_args(m)
    with DeviceContext() as ctx:
        comptime if is_defined["temperature"]():
            var temperature = Float32(
                atof(get_defined_string["temperature", "1.0"]())
            )
            bench_softmax_with_temperature_gpu[dtype, shape](
                ctx, m, "softmax_with_temperature_gpu", temperature
            )
        else:
            comptime if bench_impl_mode == 0:
                if profile_once != 0:
                    raise Error(
                        "profile-once only supports current-only or candidate-only binaries"
                    )
                bench_softmax_gpu[bench_impl_mode, dtype, shape](
                    ctx,
                    m,
                    "softmax_gpu",
                    variant,
                    replay_passes,
                    slot_filter,
                    replay_pass_filter,
                    slot_order_mode,
                )
            else:
                if profile_once != 0:
                    launch_softmax_variant_once[
                        bench_impl_mode, dtype, shape
                    ](ctx, variant)
                else:
                    bench_softmax_gpu[bench_impl_mode, dtype, shape](
                        ctx,
                        m,
                        "softmax_gpu",
                        variant,
                        replay_passes,
                        slot_filter,
                        replay_pass_filter,
                        slot_order_mode,
                    )

    if profile_once == 0:
        m.dump_report()
