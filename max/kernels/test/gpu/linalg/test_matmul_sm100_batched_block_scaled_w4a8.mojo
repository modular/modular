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
"""Batched W4A8 SM100 block-scaled matmul.

Operands are rank 3 (`(batch, rows, K)`, with B at `K/2` bytes) and scales
rank 6, so the packed FP4 weights go through a rank-3 TMA descriptor with a
nonzero batch coordinate. Each batch draws its own operands and scales, so a
batch reading its neighbour's data fails the comparison.
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
                    block_swizzle_size=8,
                    batch=2,
                    batched=True,
                ](ctx, 1000)

                run_w4a8_block_scaled_matmul[
                    N=2048,
                    K=4096,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 2, 1),
                    batch=3,
                    batched=True,
                ](ctx, 500)

                run_w4a8_block_scaled_matmul[
                    N=1024,
                    K=1152,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(cta_group, 1, 1),
                    swapAB=True,
                    batch=2,
                    batched=True,
                ](ctx, 16)

                run_w4a8_block_scaled_matmul[
                    N=2560,
                    K=8192,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 4, 1),
                    swapAB=True,
                    k_group_size=2,
                    batch=3,
                    batched=True,
                ](ctx, 100)
