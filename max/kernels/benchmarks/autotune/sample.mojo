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

from sys import env_get_dtype, env_get_int

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
)
from internal_utils import (
    Mode,
    arg_parse,
    env_get_shape,
    int_list_to_tuple,
    update_bench_config,
)


fn bench_func[
    dtype: DType, M: Int, N: Int, K: Int, stages: Int
](mut m: Bench, mode: Mode) raises:
    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            pass

        b.iter[call_fn]()

    var name = String(
        "gemm/dtype=", dtype, "/m=", M, "/n=", N, "/k=", N, "/stages=", stages
    )

    if mode == Mode.BENCHMARK:
        m.bench_function[bench_iter](BenchId(name))
    if mode == Mode.VERIFY:
        print("verifying dummy results...PASS")
    if mode == Mode.RUN:
        print("pretending to run the kernel...PASS")


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.float16]()
    alias shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
    alias shape = int_list_to_tuple[shape_int_list]()
    alias stages = env_get_int["stages", 0]()

    var runtime_x = arg_parse("x", 0)

    # define benchmark mode: [run, benchmark, verify] or a combo (run+benchmark)
    var mode = Mode(arg_parse("mode", "benchmark"))

    print("mode=" + String(mode))
    if mode == Mode.RUN:
        print("-- mode: run kernel once")
    if mode == Mode.BENCHMARK:
        print("-- mode: run kernel benchmark")
    if mode == Mode.VERIFY:
        print("-- mode: verify kernel")

    var m = Bench(
        BenchConfig(max_iters=1, max_batch_size=1, min_warmuptime_secs=0)
    )

    update_bench_config(m)

    bench_func[dtype, shape[0], shape[1], shape[2], stages](m, mode)

    m.dump_report()
