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
"""Smallest W4A8 configuration of the SM100 block-scaled matmul.

One CTA per MMA, no multicast, no AB swap: the configuration to bring up
first. M is ragged against the 128-row tile and the 128-row scale group, and
K spans nine 128-element K tiles. Both the exact and the realistic data modes
of the harness run on the same kernel instantiation.
"""

from max.gpu.host import DeviceContext
from w4a8_block_scaled_matmul_harness import run_w4a8_block_scaled_matmul


def main() raises:
    with DeviceContext() as ctx:
        run_w4a8_block_scaled_matmul[
            N=1024, K=1152, mma_n=128, cta_group=1, cluster_shape=(1, 1, 1)
        ](ctx, 1000)
        run_w4a8_block_scaled_matmul[
            N=1024,
            K=1152,
            mma_n=128,
            cta_group=1,
            cluster_shape=(1, 1, 1),
            realistic_data=True,
        ](ctx, 1000)
