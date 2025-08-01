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

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    keep,
    ThroughputMeasure,
    BenchMetric,
)
from pathlib import _dir_of_current_file


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_parsing_all_floats_in_file(
    mut b: Bencher, items_to_parse: List[String]
) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for item in items_to_parse:
            var res = atof(item)
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items_to_parse))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


fn main() raises:
    var bench = Bench()
    alias files = ["canada", "mesh"]

    @parameter
    for i in range(len(files)):
        alias filename = files[i]
        var file_path = _dir_of_current_file() / "data" / (filename + ".txt")

        var items_to_parse = file_path.read_text().splitlines()
        var nb_of_bytes = 0
        for item2 in items_to_parse:
            nb_of_bytes += len(item2)

        bench.bench_with_input[List[String], bench_parsing_all_floats_in_file](
            BenchId("atof", filename),
            items_to_parse,
            ThroughputMeasure(BenchMetric.elements, len(items_to_parse)),
            ThroughputMeasure(BenchMetric.bytes, nb_of_bytes),
        )

    print(bench)
