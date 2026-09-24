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
"""Inkling short convolution: depthwise causal conv1d with a residual.

Semantics ported from the vLLM reference's ``fused_sconv(activation=None,
use_residual=True)``: depthwise, causal, width 4, no bias, ``x + conv(x)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.state_space import short_conv_ring_commit, short_conv_ring_fwd


class ShortConvolution(Module, Shardable):
    """Depthwise causal conv1d with a residual; history lives in a
    caller-owned ring of past inputs indexed by position."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dtype: DType,
        device: DeviceRef,
        commit_conv_state: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        # False for the MTP draft, which convolves with no history.
        self.commit_conv_state = commit_conv_state
        self._sharding_strategy: ShardingStrategy | None = None
        self.weight = Weight(
            "weight", dtype, [channels, 1, kernel_size], device=device
        )

    @property
    def taps(self) -> TensorValue:
        """The weight as ``[channels, kernel_size]``."""
        return self.weight.reshape([self.channels, self.kernel_size])

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the channels across devices, or replicates them.

        Replicate suits a caller that convolves per-rank partial sums and
        reduces afterwards: the convolution is linear, so the result matches
        convolving the reduced value.
        """
        if strategy.is_replicate:
            self.weight.sharding_strategy = strategy
            self._sharding_strategy = strategy
            return
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "ShortConvolution supports the tensor parallel and replicate "
                "sharding strategies."
            )
        if self.channels % strategy.num_devices:
            raise ValueError(
                f"{self.channels} convolution channels do not divide over "
                f"{strategy.num_devices} devices"
            )
        self.weight.sharding_strategy = ShardingStrategy.rowwise(
            strategy.num_devices
        )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[ShortConvolution]:
        """Creates one per-device view of this convolution."""
        if self._sharding_strategy is None:
            raise ValueError(
                "ShortConvolution cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        replicated = self._sharding_strategy.is_replicate
        shards = []
        for device, weight in zip(
            devices, self.weight.shard(devices), strict=True
        ):
            sharded = ShortConvolution(
                channels=(
                    self.channels
                    if replicated
                    else self.channels // len(devices)
                ),
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=device,
                commit_conv_state=self.commit_conv_state,
            )
            sharded.weight = weight
            shards.append(sharded)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_ring: BufferValue,
        conv_row: TensorValue,
        input_row_offsets: TensorValue,
        positions: TensorValue,
    ) -> TensorValue:
        """Returns ``x + conv(x)``.

        Taps before the chunk read the ring slot. ``commit_conv_state``
        writes the chunk's last inputs to the ring.
        """
        out = short_conv_ring_fwd(
            x, self.taps, conv_ring, input_row_offsets, positions, conv_row
        )
        if self.commit_conv_state:
            short_conv_ring_commit(
                x, conv_ring, input_row_offsets, positions, conv_row
            )
        return out


def short_conv_ring_commit_kv(
    qkvr: TensorValue,
    k_ring: BufferValue,
    v_ring: BufferValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    k_conv_row: TensorValue,
    v_conv_row: TensorValue,
    *,
    k_col: int,
) -> None:
    """Commits the K and V conv inputs of a fused ``qkvr`` projection in one
    launch.

    Args:
        qkvr: ``[total_seq_len, q_dim + k_dim + v_dim + ...]`` projection.
        k_ring: The K site's ``[slots, ring_len, channels]`` conv state.
        v_ring: The V site's, same shape.
        input_row_offsets: ``[batch + 1]`` uint32.
        positions: ``[total_seq_len]`` uint32 position per token.
        k_conv_row: ``[batch]`` uint32 K ring slot per sequence.
        v_conv_row: ``[batch]`` uint32 V ring slot per sequence.
        k_col: First K column of ``qkvr``; V follows K.
    """
    ops.inplace_custom(
        "mo.short_conv_ring_commit_kv",
        device=qkvr.device,
        values=[
            k_ring,
            v_ring,
            qkvr,
            input_row_offsets,
            positions,
            k_conv_row,
            v_conv_row,
        ],
        out_types=[],
        parameters={"k_col": k_col},
    )


def fused_qk_rms_norm_short_conv_ragged(
    kv_params: KVCacheParams,
    qkvr: TensorValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    kv_collection: PagedCacheValues,
    q_gamma: TensorValue,
    k_gamma: TensorValue,
    k_weight: TensorValue,
    v_weight: TensorValue,
    k_conv_ring: BufferValue,
    v_conv_ring: BufferValue,
    k_conv_row: TensorValue,
    v_conv_row: TensorValue,
    log_scaling: TensorValue,
    epsilon: float,
    layer_idx: TensorValue,
    *,
    q_num_heads: int,
    apply_log_scaling: bool,
    multiply_before_cast: bool = True,
) -> TensorValue:
    """Runs a short-conv attention block's prologue in one GPU launch.

    From the fused ``qkvr`` projection: Q gets a per-head RMSNorm; K and V
    get a depthwise causal conv with residual, pre-chunk taps from their
    conv rings; K is then RMSNormed; both are stored into the paged KV
    cache. Exact for any chunk length. The rings are only read; commit them
    afterwards with :func:`short_conv_ring_commit_kv`.

    Args:
        kv_params: The KV cache parameters.
        qkvr: ``[total_tokens, q_dim + k_dim + v_dim + ...]``; columns past
            V are ignored.
        input_row_offsets: ``[batch + 1]`` uint32.
        positions: ``[total_tokens]`` uint32 position per token.
        kv_collection: The paged KV cache to store K and V into.
        q_gamma: ``[head_dim]`` query RMSNorm weight.
        k_gamma: ``[head_dim]`` key RMSNorm weight.
        k_weight: ``[kv_num_heads * head_dim, kernel_size]`` K conv taps.
        v_weight: V conv taps, same shape.
        k_conv_ring: ``[slots, ring_len, kv_num_heads * head_dim]`` K ring.
        v_conv_ring: V ring, same shape.
        k_conv_row: ``[batch]`` uint32 slot into ``k_conv_ring``.
        v_conv_row: ``[batch]`` uint32 slot into ``v_conv_ring``.
        log_scaling: ``[total_tokens]`` float32 factor each token's
            normalized query is multiplied by, after rounding to its dtype.
            Read only with ``apply_log_scaling``.
        epsilon: The RMSNorm epsilon, shared by the Q and K norms.
        layer_idx: Scalar uint32 KV-cache layer index.
        q_num_heads: Number of query heads.
        apply_log_scaling: Whether to scale Q by ``log_scaling``.
        multiply_before_cast: Whether to multiply by gamma before casting to
            the output dtype.

    Returns:
        The normalized query, ``[total_tokens, q_num_heads, head_dim]``.
    """
    return ops.inplace_custom(
        "mo.fused_qk_rms_norm_short_conv.ragged.paged",
        device=qkvr.device,
        values=[
            qkvr,
            input_row_offsets,
            positions,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            q_gamma,
            k_gamma,
            k_weight,
            v_weight,
            k_conv_ring,
            v_conv_ring,
            k_conv_row,
            v_conv_row,
            log_scaling,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=qkvr.dtype,
                shape=[qkvr.shape[0], q_num_heads, kv_params.head_dim],
                device=qkvr.device,
            )
        ],
        parameters={
            "multiply_before_cast": multiply_before_cast,
            "apply_log_scaling": apply_log_scaling,
        },
    )[0].tensor
