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

from benchmark import Bench, Bencher, BenchId
from gpu.host import DeviceContext, Dim
from layout import *


fn empty_kernel():
    pass


fn empty_kernel_many_params[
    layout_1: Layout,
    layout_2: Layout,
    layout_3: Layout,
    layout_4: Layout,
    layout_5: Layout,
    layout_6: Layout,
    layout_7: Layout,
    layout_8: Layout,
    layout_9: Layout,
]():
    pass


fn bench_empty_launch_caller(mut m: Bench, ctx: DeviceContext) raises:
    @parameter
    @always_inline
    fn bench_empty_launch(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function[empty_kernel](
                grid_dim=Dim(1), block_dim=Dim(1)
            )

        b.iter_custom[launch](ctx)

    m.bench_function[bench_empty_launch](BenchId("bench_empty_launch"))


fn bench_empty_launch_many_params_caller(
    mut m: Bench, ctx: DeviceContext
) raises:
    alias func_alias = empty_kernel_many_params[
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
        Layout([1, 2], [3, 3]),
    ]

    @parameter
    @always_inline
    fn bench_empty_launch_many_params(mut b: Bencher) raises:
        @parameter
        fn launch() raises:
            ctx.enqueue_function[func_alias](grid_dim=Dim(1), block_dim=Dim(1))

        b.iter[launch]()
        ctx.synchronize()

    m.bench_function[bench_empty_launch_many_params](
        BenchId("bench_empty_launch_many_params")
    )


def main():
    with DeviceContext() as ctx:
        var m = Bench()
        bench_empty_launch_caller(m, ctx)
        bench_empty_launch_many_params_caller(m, ctx)
        m.dump_report()
