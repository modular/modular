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

from benchmark import Bench, BenchConfig, Bencher, BenchId
from internal_utils import (
    Mode,
    arg_parse,
    env_get_shape,
    int_list_to_tuple,
    update_bench_config_args,
)

from time import sleep
from os import getenv

# mojo build sample.mojo
# mpirun -n 8 ./sample -o output.csv


fn bench_func[
    dtype: DType, M: Int, N: Int, K: Int, stages: Int
](mut m: Bench, mode: Mode, pe_rank: Int) raises:
    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher):
        @parameter
        @always_inline
        fn call_fn():
            sleep(0.01)

        b.iter[call_fn]()

    var name = String(
        "gemm/dtype=", dtype, "/m=", M, "/n=", N, "/k=", N, "/stages=", stages
    )

    if Mode.BENCHMARK == mode:
        m.bench_function[bench_iter](
            BenchId(name, input_id=String("1st-metric (pe_rank=", pe_rank, ")"))
        )
        # TODO: enable the following line after adding support for multi-output to kplot and kprofile.
        # m.bench_function[bench_iter](BenchId(name, input_id=String("2nd-metric (pe_rank=",pe_rank,")")))
    if Mode.VERIFY == mode:
        print("verifying dummy results...PASS")
    if Mode.RUN == mode:
        print("pretending to run the kernel...PASS")


def main():
    comptime dtype = env_get_dtype["dtype", DType.float16]()
    comptime shape_int_list = env_get_shape["shape", "1024x1024x1024"]()
    comptime shape = int_list_to_tuple[shape_int_list]()
    comptime stages = env_get_int["stages", 0]()

    var runtime_x = arg_parse("x", 0)

    # define benchmark mode: [run, benchmark, verify] or a combo (run+benchmark)
    var mode = Mode(arg_parse("mode", "benchmark"))

    print("mode=" + String(mode))

    if Mode.RUN == mode:
        print("-- mode: run kernel once")
    if Mode.BENCHMARK == mode:
        print("-- mode: run kernel benchmark")
    if Mode.VERIFY == mode:
        print("-- mode: verify kernel")

    var m = Bench(BenchConfig(max_iters=1, max_batch_size=1))
    var pe_rank = m.check_mpirun()
    update_bench_config_args(m)
    bench_func[dtype, shape[0], shape[1], shape[2], stages](m, mode, pe_rank)

    m.dump_report()
