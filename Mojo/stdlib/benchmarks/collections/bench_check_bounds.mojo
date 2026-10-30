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

"""Measures bounds-checked random gathers (issue #7183).

Build with ASSERT=safe (the default) or ASSERT=all. ASSERT=none is a control
that removes the bounds checks from both variants.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep


@inline(.never)
def _gather[checked: Bool](data: Span[UInt8, _], indices: Span[Int, _]) -> Int:
    var total = 0
    for i in indices:
        comptime if checked:
            total += Int(data[i])
        else:
            total += Int(data.unsafe_ptr()[unsafe_offset=i])
    return total


def bench_gather[checked: Bool](mut b: Bencher) raises:
    var data = List[UInt8](length=4096, fill=1)
    var indices = List[Int](capacity=4096)
    var x = 7
    for _ in range(4096):
        x = (x * 1103515245 + 12345) % 2147483648
        indices.append((x >> 8) % 4096)

    def gather() {imm data, imm indices}:
        keep(_gather[checked](data, indices))

    b.iter(gather)
    _ = data^
    _ = indices^


def main() raises:
    var bench = Bench(BenchConfig(min_runtime_secs=0.1, max_iters=1_000_000))
    bench.bench_function(bench_gather[True], BenchId("checked/4096"))
    bench.bench_function(bench_gather[False], BenchId("unchecked/4096"))
    bench.dump_report()
