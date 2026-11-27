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
"""Inkling short convolution and RMSNorm."""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorType, TensorValue, ops
from max.nn import Module
from max.nn._graph_ops.state_space import short_conv_ring
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.norm import RMSNorm
from max.nn.norm.rms_norm import rms_norm
from max.tensor import Tensor

_short_conv_ring = F.functional(short_conv_ring)


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
    k_conv_rows: TensorValue,
    v_conv_rows: TensorValue,
    k_layer_row: TensorValue,
    v_layer_row: TensorValue,
    log_scaling: TensorValue,
    epsilon: float,
    layer_idx: TensorValue,
    *,
    q_num_heads: int,
    apply_log_scaling: bool,
    commit: bool,
    multiply_before_cast: bool = True,
) -> TensorValue:
    """Runs a short-conv attention block's prologue in one GPU launch.

    From the fused ``qkvr`` projection: Q gets a per-head RMSNorm; K and V
    get a depthwise causal conv with residual, pre-chunk taps from their
    conv rings; K is then RMSNormed; both are stored into the paged KV
    cache. Exact for any chunk length. With ``commit``, each sequence's last
    ``ring_len`` K and V conv inputs are written into the rings in the same
    launch; without it the rings are only read.

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
        k_conv_rows: ``[num_layers, batch]`` uint32 slot into ``k_conv_ring``.
        v_conv_rows: ``[num_layers, batch]`` uint32 slot into ``v_conv_ring``.
        k_layer_row: Scalar uint32 CPU row of ``k_conv_rows`` this layer reads.
        v_layer_row: The same for ``v_conv_rows``.
        log_scaling: ``[total_tokens]`` float32 factor each token's
            normalized query is multiplied by, after rounding to its dtype.
            Read only with ``apply_log_scaling``.
        epsilon: The RMSNorm epsilon, shared by the Q and K norms.
        layer_idx: Scalar uint32 KV-cache layer index.
        q_num_heads: Number of query heads.
        apply_log_scaling: Whether to scale Q by ``log_scaling``.
        commit: Whether to commit the K and V conv inputs to their rings.
        multiply_before_cast: Whether to multiply by gamma before casting to
            the output dtype.

    Returns:
        The normalized query, ``[total_tokens, q_num_heads, head_dim]``.
    """
    return ops.inplace_custom(
        "kv_cache.fused_qk_rms_norm_short_conv.ragged.paged",
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
            k_conv_rows,
            v_conv_rows,
            k_layer_row,
            v_layer_row,
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
            "commit": commit,
        },
    )[0].tensor


class InklingRMSNorm(RMSNorm):
    """RMSNorm that scales before casting back, as max.nn.legacy.RMSNorm does."""

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMSNorm, scaling by the weight before the cast back."""
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)


class ShortConvolution(Module[..., Tensor]):
    """Depthwise causal conv1d with a residual.

    History lives in a caller-owned ring of past inputs indexed by position.
    Each device convolves its own shard of the input with its own shard of
    the weight, so a replicated weight over per-device partial sums gives
    partial sums of the convolution.
    """

    def __init__(
        self, *, channels: int, kernel_size: int, commit_conv_state: bool = True
    ) -> None:
        """Stores the conv shape and allocates a zero tap weight.

        ``commit_conv_state`` selects whether ``forward`` writes the new
        inputs into the caller's ring.
        """
        self.channels = channels
        self.kernel_size = kernel_size
        self.commit_conv_state = commit_conv_state
        self.weight = Tensor.zeros([channels, 1, kernel_size])

    @property
    def taps(self) -> Tensor:
        """The weight as ``[channels, kernel_size]``."""
        return self.weight.reshape([self.channels, self.kernel_size])

    def forward(
        self,
        x: Tensor,
        conv_ring: Tensor,
        conv_rows: Tensor,
        layer_row: Tensor,
        input_row_offsets: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Returns ``x + conv(x)``, with no known placement."""
        return _short_conv_ring(
            x,
            self.taps,
            conv_ring,
            input_row_offsets,
            positions,
            conv_rows,
            layer_row,
            commit=self.commit_conv_state,
        )
