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
"""W4A8 SM100 block-scaled matmul with weight tiles prefetched before PDL.

With `prefetch_tiles_n = 2` the producer loads the first weight tiles ahead
of the programmatic-dependent-launch wait and completes the activation tiles
afterwards, so the weight and activation halves of one pipeline stage arrive
through separate TMA issues. Each half must arm the stage barrier with its
own global-memory byte count: the packed FP4 weights with half their
shared-memory footprint, the E4M3 activations with all of it. A wrong split
either hangs or releases a stage early.

Covers 1 and 2 CTAs per MMA, with and without AB swap (which moves the
weights into the A slot), K = 128: a single K tile, fewer than the
prefetch depth, and `k_group_size = 2`, where each prefetched stage carries
two K tiles.
"""

from max.gpu.host import DeviceContext
from w4a8_block_scaled_matmul_harness import run_w4a8_block_scaled_matmul


def main() raises:
    with DeviceContext() as ctx:
        comptime for mma_n in [64, 128, 192, 256]:
            run_w4a8_block_scaled_matmul[
                N=1024,
                K=1152,
                mma_n=mma_n,
                cta_group=1,
                cluster_shape=(1, 1, 1),
                block_swizzle_size=8,
                prefetch_tiles_n=2,
            ](ctx, 1000)

            run_w4a8_block_scaled_matmul[
                N=1024,
                K=1152,
                mma_n=mma_n,
                cta_group=1,
                cluster_shape=(1, 1, 1),
                swapAB=True,
                prefetch_tiles_n=2,
            ](ctx, 16)

        comptime for mma_n in [128, 256]:
            run_w4a8_block_scaled_matmul[
                N=1024,
                K=1152,
                mma_n=mma_n,
                cta_group=2,
                cluster_shape=(2, 1, 1),
                prefetch_tiles_n=2,
            ](ctx, 1000)

            run_w4a8_block_scaled_matmul[
                N=2560,
                K=8192,
                mma_n=mma_n,
                cta_group=2,
                cluster_shape=(4, 4, 1),
                prefetch_tiles_n=2,
            ](ctx, 777)

            run_w4a8_block_scaled_matmul[
                N=256,
                K=128,
                mma_n=mma_n,
                cta_group=2,
                cluster_shape=(8, 2, 1),
                prefetch_tiles_n=2,
            ](ctx, 999)

            run_w4a8_block_scaled_matmul[
                N=1024,
                K=1152,
                mma_n=mma_n,
                cta_group=2,
                cluster_shape=(2, 1, 1),
                swapAB=True,
                prefetch_tiles_n=2,
            ](ctx, 16)

        comptime for swap in [False, True]:
            run_w4a8_block_scaled_matmul[
                N=1024,
                K=4096,
                mma_n=128,
                cta_group=1,
                cluster_shape=(1, 1, 1),
                swapAB=swap,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, 16 if swap else 1000)

            run_w4a8_block_scaled_matmul[
                N=2048,
                K=4096,
                mma_n=256,
                cta_group=2,
                cluster_shape=(4, 2, 1),
                swapAB=swap,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, 100 if swap else 500)

            run_w4a8_block_scaled_matmul[
                N=2560,
                K=8192,
                mma_n=128,
                cta_group=2,
                cluster_shape=(4, 4, 1),
                swapAB=swap,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, 100 if swap else 777)
