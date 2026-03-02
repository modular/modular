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

"""Expert Parallelism (EP) Communication Configuration."""

from dataclasses import dataclass

from max.dtype import DType
from max.nn.float8_config import Float8Config

# We always use two groups of SHMEM shared memory buffers to avoid race
# conditions between the dispatch and combine phases of Expert Parallelism
# communication.
#
# Expert Parallelism consists of two sequential phases, each with two kernels:
# 1. Dispatch phase: Route tokens to experts on different GPUs
#    - dispatch_async_kernel: Initiates token transfer
#    - dispatch_wait_kernel: Ensures current GPU received all tokens from other
#      GPUs
# 2. Combine phase: Return expert outputs to original GPUs
#    - combine_async_kernel: Initiates expert output transfer
#    - combine_wait_kernel: Ensures current GPU received all expert outputs
#
# When dispatch_wait_kernel completes on the current GPU, it only guarantees that
# THIS GPU has received all tokens. Other GPUs may still be writing to their
# dispatch buffers or reading their own receive buffers. Therefore, we cannot
# safely reuse dispatch phase buffers for the combine phase.
#
# We use dedicated buffers for each phase. When combine_wait_kernel completes, it
# guarantees all GPUs have at least started their combine_async_kernel, which implies
# they've finished their dispatch_wait_kernel. Only then is it safe to reuse the
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

    node_id: int = -1
    """ID of the current node. Will be set by the EPCommInitializer."""

    dispatch_fp8_config: Float8Config | None = None
    """Float8 configuration used for dispatch token quantization."""

    fused_shared_expert: bool = False
    """Whether to fuse the shared expert computation with the routed experts."""

    def estimate_memory_usage(self) -> int:
        """Estimate the EP communication memory usage per device per buffer group.

        Returns:
            Estimated memory usage in bytes per device per buffer group.
        """
        return estimate_ep_memory_usage(
            hidden_size=self.hidden_size,
            dispatch_dtype=self.dispatch_dtype,
            combine_dtype=self.combine_dtype,
            max_tokens_per_rank=self.max_tokens_per_rank,
            n_experts=self.n_experts,
            top_k=self.top_k,
        )

    def __post_init__(self):
        if self.dispatch_dtype != DType.bfloat16:
            if self.dispatch_fp8_config is None:
                raise ValueError(
                    "dispatch_fp8_config must be specified when dispatch_dtype is not bfloat16"
                )

            if self.dispatch_dtype.is_float8():
                if not self.dispatch_fp8_config.input_scale.is_block:
                    raise NotImplementedError(
                        "Only block-wise quantization is supported for float8_e4m3fn and float8_e4m3fnuz"
                    )

            elif self.dispatch_dtype in (DType.uint8, DType.float4_e2m1fn):
                if not self.dispatch_fp8_config.is_nvfp4:
                    raise ValueError(
                        "dispatch_fp8_config must be an NVFP4 configuration when dispatch_dtype is uint8 or float4_e2m1fn"
                    )

            else:
                raise ValueError(
                    f"Unsupported dispatch dtype: {self.dispatch_dtype}"
                )


def estimate_ep_memory_usage(
    *,
    hidden_size: int,
    dispatch_dtype: DType,
    combine_dtype: DType,
    max_tokens_per_rank: int,
    n_experts: int,
    top_k: int,
) -> int:
    """Estimate the EP communication memory usage per device per buffer group.

    This is a standalone function so it can be called both from
    :class:`EPCommInitializer` (which has a fully-validated ``EPConfig``)
    and from memory estimators that only need the numeric fields.

    Args:
        hidden_size: Model hidden dimension.
        dispatch_dtype_size: Size in bytes of the dispatch data type.
        combine_dtype_size: Size in bytes of the combine data type.
        max_tokens_per_rank: Maximum tokens per GPU rank.
        n_experts: Total number of routed experts.
        top_k: Number of experts each token is routed to.

    Returns:
        Total estimated memory usage in bytes per device per buffer group.
    """

    def _n_elems_to_bytes(dtype: DType, n_elems: int) -> int:
        if dtype in (DType.uint8, DType.float4_e2m1fn):
            # Account for the scales. For NVFP4 format, every 16 FP4 elements
            # share one FP8 scale factor. The size of the scales is one eighth
            # of the size of the FP4 quants (8 bits / (16 * 4 bits)).
            return int(n_elems // 2 * dtype.size_in_bytes * 1.125)
        else:
            return n_elems * dtype.size_in_bytes

    d_token_size = _n_elems_to_bytes(dispatch_dtype, hidden_size)
    dispatch_send_buf_size = max_tokens_per_rank * d_token_size
    dispatch_recv_buf_size = n_experts * max_tokens_per_rank * d_token_size

    c_token_size = hidden_size * combine_dtype.size_in_bytes
    combine_send_buf_size = n_experts * max_tokens_per_rank * c_token_size
    combine_recv_buf_size = top_k * max_tokens_per_rank * c_token_size

    return (
        dispatch_send_buf_size
        + dispatch_recv_buf_size
        + combine_send_buf_size
        + combine_recv_buf_size
    )
