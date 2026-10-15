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
"""W4A8 SM100 block-scaled matmul across CTA groups, MMA N, and clusters.

Mirrors the MXFP4/NVFP4 sweep in `test_matmul_sm100_block_scaled_fp4`, with
K rounded to the multiple of 128 the packed FP4 TMA copy requires:

- 1 and 2 CTAs per MMA crossed with every supported MMA N
- multicast clusters (4, 2, 1) with two K tiles per stage, and (4, 4, 1)
  with `alpha = 0.5`
- a single-row decode shape
- the fused elementwise epilogue
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
                ](ctx, 1000)

                run_w4a8_block_scaled_matmul[
                    N=2048,
                    K=4096,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 2, 1),
                    k_group_size=2,
                ](ctx, 500)

                run_w4a8_block_scaled_matmul[
                    N=2560,
                    K=8192,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 4, 1),
                    block_swizzle_size=1,
                ](ctx, 777, alpha=0.5)

                run_w4a8_block_scaled_matmul[
                    N=576,
                    K=7168,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(4, 4, 1),
                    block_swizzle_size=1,
                ](ctx, 1, alpha=0.5)

        comptime for cta_group in [1, 2]:
            comptime for mma_n in [64, 128]:
                run_w4a8_block_scaled_matmul[
                    N=1024,
                    K=1152,
                    mma_n=mma_n,
                    cta_group=cta_group,
                    cluster_shape=(cta_group, 1, 1),
                    block_swizzle_size=8,
                    fused_epilogue=True,
                ](ctx, 16)
