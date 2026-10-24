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
"""W4A8 SM100 block-scaled matmul with A and B swapped.

Swapping computes `C^T = B A^T`, so the packed E2M1 weights occupy the MMA's
A slot and the E4M3 activations its B slot: the operand-type, TMA, and
transaction-byte handling must follow the operand, not the slot. Covers:

- small M across CTA groups and MMA N, with and without multicast
- the Llama 3.1 405B TP8 shapes with their tuned configurations (M = 128;
  smaller M makes the swapped N too small for the TMA box)
- swapAB combined with the fused elementwise epilogue
- realistic data with a non-power-of-two `alpha`, where the swapped
  accumulation order differs from the reference's
"""

from max.gpu.host import DeviceContext
from w4a8_block_scaled_matmul_harness import run_w4a8_block_scaled_matmul


def main() raises:
    with DeviceContext() as ctx:
        comptime for cta_group in [1, 2]:
            comptime for mma_n in [64, 128, 192, 256]:
                run_w4a8_block_scaled_matmul[
                    N=1024,
                    K=1152,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(cta_group, 1, 1),
                    swapAB=True,
                ](ctx, 16)

                run_w4a8_block_scaled_matmul[
                    N=2560,
                    K=8192,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 4, 1),
                    swapAB=True,
                    k_group_size=2,
                ](ctx, 100)

        comptime for n in [2304, 6656, 13312]:
            run_w4a8_block_scaled_matmul[
                N=n,
                K=16384,
                mma_n=64,
                cta_group=2,
                cluster_shape=(4, 1, 1),
                swapAB=True,
                num_accum_pipeline_stages=1,
                num_clc_pipeline_stages=0,
            ](ctx, 128)

        run_w4a8_block_scaled_matmul[
            N=7168,
            K=16384,
            mma_n=64,
            cta_group=2,
            cluster_shape=(2, 1, 1),
            swapAB=True,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
        ](ctx, 128)

        run_w4a8_block_scaled_matmul[
            N=1024,
            K=1152,
            mma_n=64,
            cta_group=1,
            cluster_shape=(1, 1, 1),
            swapAB=True,
            fused_epilogue=True,
        ](ctx, 16)

        run_w4a8_block_scaled_matmul[
            N=1024,
            K=1152,
            mma_n=64,
            cta_group=1,
            cluster_shape=(1, 1, 1),
            swapAB=True,
            realistic_data=True,
        ](ctx, 16, alpha=0.3)

        run_w4a8_block_scaled_matmul[
            N=2560,
            K=8192,
            mma_n=128,
            cta_group=2,
            cluster_shape=(4, 4, 1),
            swapAB=True,
            k_group_size=2,
            realistic_data=True,
        ](ctx, 100, alpha=0.3)
