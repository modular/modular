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

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from memory import LegacyUnsafePointer as UnsafePointer


@parameter
fn bench_allocation(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(10000):
            var a = UnsafePointer[Int].alloc(100)
            keep(a)
            a.free()

    b.iter[call_fn]()


def main():
    var m = Bench(BenchConfig())
    m.bench_function[bench_allocation](BenchId("bench_allocation"))
    m.dump_report()
