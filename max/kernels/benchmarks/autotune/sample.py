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

from bencher_utils import Bench, ThroughputMeasure, arg_parse


def main() -> None:
    shape = arg_parse("shape", "1x1x1")
    M, N, K = [int(v) for v in shape.split("x")]
    dtype = arg_parse("dtype", "DType.float16")
    stages = int(arg_parse("stages", "0"))

    output_path = arg_parse("output", "output.csv", short_handle="o")

    flops = ThroughputMeasure(Bench.flops, int(2 * M * N * K))
    b = Bench(
        f"gemm/dtype={dtype}/m={M}/n={N}/k={K}/stages={stages}",
        iters=1,
        met=0.1,
        metric_list=[flops],
    )
    b.dump_report(output_path=output_path)


if __name__ == "__main__":
    main()
