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

"""Expert Parallelism (EP) Communication Configuration."""

from dataclasses import dataclass

from max.dtype import DType
from max.nn.float8_config import Float8InputScaleSpec

# We always use two groups of SHMEM shared memory buffers to avoid race
# conditions between the dispatch and combine phases of Expert Parallelism
# communication.
#
# Expert Parallelism consists of two sequential phases, each with two kernels:
# 1. Dispatch phase: Route tokens to experts on different GPUs
#    - dispatch_kernel: Initiates token transfer
#    - dispatch_cb_kernel: Ensures current GPU received all tokens from other
#      GPUs
# 2. Combine phase: Return expert outputs to original GPUs
#    - combine_kernel: Initiates expert output transfer
#    - combine_cb_kernel: Ensures current GPU received all expert outputs
#
# When dispatch_cb_kernel completes on the current GPU, it only guarantees that
# THIS GPU has received all tokens. Other GPUs may still be writing to their
# dispatch buffers or reading their own receive buffers. Therefore, we cannot
# safely reuse dispatch phase buffers for the combine phase.
#
# We use dedicated buffers for each phase. When combine_cb_kernel completes, it
# guarantees all GPUs have at least started their combine_kernel, which implies
# they've finished their dispatch_cb_kernel. Only then is it safe to reuse the
# dispatch buffers for the next iteration.
NUM_GROUPS = 2


@dataclass
class EPConfig:
    """Configuration for Expert Parallelism (EP) communication."""

    dispatch_dtype: DType
    """Data type used for dispatching tokens to experts."""

    combine_dtype: DType
    """Data type used for combining expert outputs."""

    hidden_size: int
    """Size of the hidden dimension in the model."""

    top_k: int
    """Number of top experts to route each token to."""

    n_experts: int
    """Total number of experts in the model."""

    max_tokens_per_rank: int
    """Maximum number of tokens processed per GPU rank."""

    n_gpus_per_node: int
    """Number of GPUs available per node."""

    n_nodes: int
    """Total number of nodes in the distributed setup."""

    dispatch_fp8_config: Float8InputScaleSpec | None = None
    """Configuration for float8 quantization of the dispatch tokens."""

    def __post_init__(self):
        if self.dispatch_dtype.is_float8():
            if self.dispatch_fp8_config is None:
                raise ValueError(
                    "dispatch_fp8_config must be specified when dispatch_dtype is float8_e4m3fn or float8_e4m3fnuz"
                )

            if not self.dispatch_fp8_config.is_block:
                raise NotImplementedError(
                    "Only block-wise quantization is supported for float8_e4m3fn and float8_e4m3fnuz"
                )
