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

from complex import ComplexSIMD
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from random import seed
from sys import env_get_string, env_get_int, env_get_dtype
from internal_utils import env_get_shape, int_list_to_tuple

from nn.fft.fft import (
    fft,
    _intra_block_fft_kernel_radix_n,
    _get_ordered_bases_processed_list,
)

comptime _TestValues[complex_dtype: DType] = List[
    Tuple[List[Int], List[ComplexSIMD[complex_dtype, 1]]]
]


fn _get_test_values_128[
    complex_dtype: DType
](out res: _TestValues[complex_dtype]):
    """Get the series on the lhs, and the expected complex values on rhs.

    Notes:
        These values are only for testing against series with length 128.
    """
    comptime Complex = ComplexSIMD[complex_dtype, 1]
    # fmt: off
    res = [
        (
            List(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            List(
                Complex(64, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(64, 0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0), Complex(0),
                Complex(0), Complex(0), Complex(0), Complex(0),
            ),
        ),
        (
            List(13, 31, 51, 43, 41, 28, 98, 40, 10, 2, 29, 11, 87, 90, 78, 73,
            1, 22, 48, 15, 63, 49, 71, 16, 74, 61, 91, 79, 12, 100, 92, 6,
            21, 72, 45, 30, 62, 97, 23, 60, 8, 9, 86, 53, 75, 70, 25, 50,
            20, 96, 27, 83, 88, 76, 82, 42, 89, 69, 94, 38, 33, 35, 17, 14,
            26, 67, 99, 32, 95, 44, 64, 36, 66, 57, 37, 93, 19, 81, 54, 7,
            59, 3, 58, 34, 46, 77, 80, 47, 18, 85, 68, 84, 65, 5, 39, 4,
            52, 56, 55, 24, 80, 100, 91, 31, 79, 56, 41, 1, 87, 68, 81, 83,
            55, 98, 69, 82, 25, 43, 66, 86, 8, 84, 2, 34, 65, 73, 57, 29),
            List(
                Complex(6724.0, 0.0), Complex(-156.29818818, 100.55380564),
                Complex(-154.46186294, 358.35832448), Complex(-316.70533589, 68.87259552),
                Complex(-235.36501962, 168.72629575), Complex(153.36419219, -92.41119721),
                Complex(-167.97031692, -297.19454317), Complex(129.45766604, 252.75436499),
                Complex(-276.25393942, -143.20742064), Complex(473.81747787, 144.00393585),
                Complex(320.84179105, 25.17023746), Complex(-158.88631562, 220.18592989),
                Complex(-301.98202028, -497.41277706), Complex(-303.80261787, -48.39312759),
                Complex(-31.01725790, -135.61181326), Complex(-25.07873719, 256.84842535),
                Complex(-541.98989873, 157.59797974), Complex(59.92361411, 633.35206540),
                Complex(178.19036588, 147.50626608), Complex(-198.01406228, 134.93497624),
                Complex(271.22992526, -58.42798577), Complex(169.03650037, 21.35203365),
                Complex(-367.91968034, -147.58223984), Complex(-314.27893266, -225.99217847),
                Complex(-2.47523190, -166.68442802), Complex(-152.65863234, -53.19315472),
                Complex(-204.04087748, -159.01146349), Complex(220.66531481, -369.42260824),
                Complex(-71.35655134, -37.13141518), Complex(-68.56609846, 293.66828693),
                Complex(-61.10304087, 5.44109104), Complex(-97.66658182, -62.90052013),
                Complex(-376.0, -544.0), Complex(-138.48726553, 159.68362255),
                Complex(76.08025032, 328.85094309), Complex(-14.20841142, 57.48766158),
                Complex(-84.53504416, -290.91404046), Complex(13.63508856, 10.08330984),
                Complex(150.11572968, 91.71350905), Complex(431.86537249, 229.85517425),
                Complex(-26.50543210, -88.35138354), Complex(-206.02323494, 264.85461217),
                Complex(177.55994584, 238.77452385), Complex(407.19246988, -35.05565566),
                Complex(-204.32192289, -106.42113879), Complex(-37.18822859, -219.42794172),
                Complex(-118.71555940, 173.21844419), Complex(-398.99093613, -76.39699575),
                Complex(-146.01010126, -78.40202025), Complex(-32.24435434, -86.59838054),
                Complex(-157.26272934, -267.70569965), Complex(-13.92019860, 183.79322094),
                Complex(247.72582152, -19.97028849), Complex(437.03364716, -451.81031090),
                Complex(-464.63944250, 388.84686941), Complex(35.83260476, -441.59172457),
                Complex(-114.76539655, 163.12562383), Complex(-8.35436859, 199.11883066),
                Complex(317.99397617, -39.65937037), Complex(142.14939725, 180.29153298),
                Complex(194.60481151, 232.37931206), Complex(-379.57004602, 371.85170987),
                Complex(-37.65129124, 313.45244404), Complex(-69.03079898, -118.97609904),
                Complex(196.0, 0.0), Complex(-69.03079898, 118.97609904),
                Complex(-37.65129124, -313.45244404), Complex(-379.57004602, -371.85170987),
                Complex(194.60481151, -232.37931206), Complex(142.14939725, -180.29153298),
                Complex(317.99397617, 39.65937037), Complex(-8.35436859, -199.11883066),
                Complex(-114.76539655, -163.12562383), Complex(35.83260476, 441.59172457),
                Complex(-464.63944250, -388.84686941), Complex(437.03364716, 451.81031090),
                Complex(247.72582152, 19.97028849), Complex(-13.92019860, -183.79322094),
                Complex(-157.26272934, 267.70569965), Complex(-32.24435434, 86.59838054),
                Complex(-146.01010126, 78.40202025), Complex(-398.99093613, 76.39699575),
                Complex(-118.71555940, -173.21844419), Complex(-37.18822859, 219.42794172),
                Complex(-204.32192289, 106.42113879), Complex(407.19246988, 35.05565566),
                Complex(177.55994584, -238.77452385), Complex(-206.02323494, -264.85461217),
                Complex(-26.50543210, 88.35138354), Complex(431.86537249, -229.85517425),
                Complex(150.11572968, -91.71350905), Complex(13.63508856, -10.08330984),
                Complex(-84.53504416, 290.91404046), Complex(-14.20841142, -57.48766158),
                Complex(76.08025032, -328.85094309), Complex(-138.48726553, -159.68362255),
                Complex(-376.0, 544.0), Complex(-97.66658182, 62.90052013),
                Complex(-61.10304087, -5.44109104), Complex(-68.56609846, -293.66828693),
                Complex(-71.35655134, 37.13141518), Complex(220.66531481, 369.42260824),
                Complex(-204.04087748, 159.01146349), Complex(-152.65863234, 53.19315472),
                Complex(-2.47523190, 166.68442802), Complex(-314.27893266, 225.99217847),
                Complex(-367.91968034, 147.58223984), Complex(169.03650037, -21.35203365),
                Complex(271.22992526, 58.42798577), Complex(-198.01406228, -134.93497624),
                Complex(178.19036588, -147.50626608), Complex(59.92361411, -633.35206540),
                Complex(-541.98989873, -157.59797974), Complex(-25.07873719, -256.84842535),
                Complex(-31.01725790, 135.61181326), Complex(-303.80261787, 48.39312759),
                Complex(-301.98202028, 497.41277706), Complex(-158.88631562, -220.18592989),
                Complex(320.84179105, -25.17023746), Complex(473.81747787, -144.00393585),
                Complex(-276.25393942, 143.20742064), Complex(129.45766604, -252.75436499),
                Complex(-167.97031692, 297.19454317), Complex(153.36419219, 92.41119721),
                Complex(-235.36501962, -168.72629575), Complex(-316.70533589, -68.87259552),
                Complex(-154.46186294, -358.35832448), Complex(-156.29818818, -100.55380564)
            ),
        ),
    ]
    # fmt: on


def _bench_sequential_intra_block_fft_launch_radix_n[
    in_dtype: DType,
    out_dtype: DType,
    in_layout: Layout,
    out_layout: Layout,
    *,
    bases: List[UInt],
](
    output: LayoutTensor[mut=True, out_dtype, out_layout],
    x: LayoutTensor[mut=False, in_dtype, in_layout],
    ctx: DeviceContext,
    mut b: Bencher,
):
    comptime length = UInt(in_layout.shape[1].value())
    comptime bases_processed = _get_ordered_bases_processed_list[
        length, bases, "gpu"
    ]()
    comptime ordered_bases = bases_processed[0]
    comptime processed_list = bases_processed[1]

    @parameter
    fn _calc_total_offsets() -> Tuple[UInt, List[UInt]]:
        comptime last_base = ordered_bases[len(ordered_bases) - 1]
        var bases = materialize[ordered_bases]()
        var c = Int((length // last_base) * (last_base - 1)) * len(bases)
        var offsets = List[UInt](capacity=c)
        var val = UInt(0)
        for base in bases:
            offsets.append(val)
            val += (length // base) * (base - 1)
        return val, offsets^

    comptime total_offsets = _calc_total_offsets()
    comptime total_twfs = total_offsets[0]
    comptime twf_offsets = total_offsets[1]
    comptime num_threads = length // ordered_bases[len(ordered_bases) - 1]

    @parameter
    fn call_fn[
        in_dtype: DType,
        out_dtype: DType,
        in_layout: Layout,
        out_layout: Layout,
        in_origin: ImmutOrigin,
        out_origin: MutOrigin,
        *,
        ordered_bases: List[UInt],
        processed_list: List[UInt],
    ](
        output: LayoutTensor[out_dtype, out_layout, out_origin],
        x: LayoutTensor[in_dtype, in_layout, in_origin],
    ):
        for _ in range(10_000):
            _intra_block_fft_kernel_radix_n[
                in_dtype,
                out_dtype,
                in_layout,
                out_layout,
                length=length,
                ordered_bases=ordered_bases,
                processed_list=processed_list,
                do_rfft=True,
                inverse=False,
                total_twfs=total_twfs,
                twf_offsets=twf_offsets,
                warp_exec = UInt(32) >= num_threads,
            ](output, x)

    comptime func = call_fn[
        in_dtype,
        out_dtype,
        in_layout,
        out_layout,
        x.origin,
        output.origin,
        ordered_bases=ordered_bases,
        processed_list=processed_list,
    ]
    ctx.enqueue_function_checked[func, func](
        output, x, grid_dim=1, block_dim=Int(num_threads)
    )
    ctx.synchronize()
    _ = processed_list  # origin bug


@parameter
fn bench_sequential_intra_block_radix_n_rfft[
    dtype: DType, bases: List[UInt], test_values: _TestValues[dtype]
](mut b: Bencher) raises:
    comptime values = test_values[len(test_values) - 1]
    comptime SIZE = len(values[0])
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(1, SIZE, 1)
    comptime out_layout = Layout.row_major(1, SIZE, 2)
    comptime calc_dtype = dtype
    comptime Complex = ComplexSIMD[calc_dtype, 1]

    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[out_dtype](out_layout.size())
        out.enqueue_fill(0)
        var x = ctx.enqueue_create_buffer[in_dtype](in_layout.size())
        x.enqueue_fill(0)
        ref series = values[0]
        with x.map_to_host() as x_host:
            for i in range(SIZE):
                x_host[i] = series[i]

        var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
            out.unsafe_ptr()
        )
        var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](
            x.unsafe_ptr()
        )

        @always_inline
        @parameter
        fn call_fn(ctx: DeviceContext) raises:
            _bench_sequential_intra_block_fft_launch_radix_n[bases=bases](
                out_tensor, x_tensor, ctx, b
            )

        b.iter_custom[call_fn](ctx)

        _ = out_tensor
        _ = x_tensor


@parameter
fn bench_cpu_radix_n_rfft[
    dtype: DType,
    bases: List[UInt],
    batches: UInt,
    test_values: _TestValues[dtype],
    *,
    cpu_workers: Optional[UInt] = None,
](mut b: Bencher) raises:
    comptime values = test_values[len(test_values) - 1]
    comptime SIZE = len(values[0])
    comptime BATCHES = batches
    comptime in_dtype = dtype
    comptime out_dtype = dtype
    comptime in_layout = Layout.row_major(Int(BATCHES), Int(SIZE), 1)
    comptime out_layout = Layout.row_major(Int(BATCHES), Int(SIZE), 2)
    comptime calc_dtype = dtype
    comptime Complex = ComplexSIMD[calc_dtype, 1]

    var out = List[Scalar[out_dtype]](capacity=out_layout.size())
    var x = List[Scalar[in_dtype]](capacity=in_layout.size())
    ref series = values[0]
    var idx = 0
    for _ in range(BATCHES):
        for i in range(SIZE):
            x[idx] = series[i]
            idx += 1

    var out_tensor = LayoutTensor[mut=True, out_dtype, out_layout](
        out.unsafe_ptr()
    )
    var x_tensor = LayoutTensor[mut=False, in_dtype, in_layout](x.unsafe_ptr())

    @always_inline
    @parameter
    fn call_fn() raises:
        fft[bases=bases, target="cpu"](
            out_tensor, x_tensor, DeviceContext(), cpu_workers=cpu_workers
        )

    b.iter[call_fn]()

    _ = out_tensor
    _ = x_tensor


fn _get_bases(out res: List[UInt]):
    var bases_int = env_get_shape["shape", "16x8"]()
    res = {capacity = len(bases_int)}
    for b in bases_int:
        res.append(UInt(b))


def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime name = env_get_string["name", "cpu_multi_threaded"]()
    comptime dtype = env_get_dtype["dtype", DType.float64]()
    comptime bases = _get_bases()
    comptime test_values = _get_test_values_128[dtype]()
    comptime b = bases.__str__().replace("UInt(", "").replace(")", "")

    comptime cpu_bench = "bench_cpu_radix_n_rfft["
    comptime cpu_fn = bench_cpu_radix_n_rfft
    comptime N = 1_000_000

    @parameter
    if name == "gpu_sequential":
        comptime func = bench_sequential_intra_block_radix_n_rfft
        comptime func_name = "bench_sequential_intra_block_radix_n_rfft["
        m.bench_function[func[dtype, bases, test_values]](
            BenchId(String(func_name, b, ", 128]"))
        )
    elif name == "cpu_single_threaded":
        m.bench_function[
            cpu_fn[dtype, bases, N, test_values, cpu_workers = UInt(1)]
        ](BenchId(String(cpu_bench, b, ", 1_000_000, 128, workers=1]")))
    elif name == "cpu_multi_threaded":
        m.bench_function[cpu_fn[dtype, bases, N, test_values]](
            BenchId(String(cpu_bench, b, ", 1_000_000, 128, workers=n]"))
        )

    results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0].__round__(3), sep=", ")
