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

"""Qwen3VL text rotary embedding."""

from __future__ import annotations

from functools import cached_property

from max.dtype import DType
from max.graph import (
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)


class Qwen3VLTextRotaryEmbedding(Llama3RotaryEmbedding):
    """
    RotaryEmbedding for Qwen3VL text model with interleaved MRoPE support.

    This implementation handles 3D position IDs (T, H, W dimensions) and applies
    interleaved multi-axis rotary position embedding (MRoPE) to reorganize frequency
    layout from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...TT].
    """

    mrope_section: list[int]
    """The number of tokens in each dimension of the 3D position IDs [T, H, W]."""
    _inv_freqs: TensorValue | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        dtype: DType,
        mrope_section: list[int],
        head_dim: int | None = None,
        _freqs_cis: TensorValueLike | None = None,
        interleaved: bool = True,
        scaling_params: Llama3RopeScalingParams | None = None,
    ) -> None:
        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            head_dim,
            _freqs_cis,
            interleaved,
            scaling_params,
        )
        self.mrope_section = mrope_section
        self.dtype = dtype

    @cached_property
    def inv_freqs(self) -> TensorValue:
        if self._inv_freqs is None:
            # Match HuggingFace: compute in float32, round to bfloat16 (like parameter storage), then back to float32
            self._inv_freqs = (
                self._compute_inv_freqs().cast(self.dtype).cast(DType.float32)
            )
        return self._inv_freqs

    def freqs_cis_base_position_ids(
        self, position_ids: TensorValue
    ) -> TensorValue:
        """
        Computes the frequency tensor for complex exponentials (cis)
        for a given seq_len. Tensor is scaled with theta parameter.
        Required to apply Rotary Position Embedding (RoPE) to tensor.
        See 'Roformer: Enhanced Transformer with Rotary Embedding'
        (arxiv.org/pdf/2104.09864).

        Args:
            position_ids: (3, total_seq_len).

        Returns:
            The frequency tensor for complex exponentials with shape (total_seq_len, head_dim / 2, 2)
        """
        inv_freqs = self.inv_freqs  # [head_dim // 2]
        inv_freq = ops.unsqueeze(
            ops.unsqueeze(inv_freqs, 0), 0
        )  # [1, 1, head_dim // 2]
        position_ids = ops.unsqueeze(position_ids, -1).cast(
            DType.float32
        )  # [3, total_seq_len, 1]

        # Rotation matrix for block i =  [cos(m*theta_i) -sin(m*theta_i); sin(m*theta_i) -cos(m*theta_i)] for each position_id m.
        freqs = position_ids * inv_freq  # [3, total_seq_len, head_dim // 2]
        freqs = self._apply_interleaved_mrope(
            freqs, self.mrope_section, position_ids
        )  # [total_seq_len, head_dim // 2]
        self._freqs_cis = ops.stack(
            [ops.cos(freqs), ops.sin(freqs)], axis=-1
        ).cast(self.dtype)  # [total_seq_len, head_dim // 2, 2]
        return TensorValue(self._freqs_cis)

    def _apply_interleaved_mrope(
        self,
        freqs: TensorValue,
        mrope_section: list[int],
        position_ids: TensorValue,
    ) -> TensorValue:
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            position_ids: (3, total_seq_len, 1)
            freqs: (3, total_seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            freqs_t: (total_seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T

        # Create indices for H dimension (offset=1, step=3)
        # These indices select positions [1, 4, 7, ...] in the head_dim dimension
        h_offset = 1
        step = 3
        h_length = mrope_section[h_offset] * step
        h_indices = ops.range(
            h_offset,
            h_length,
            step,
            device=position_ids.device,
            dtype=DType.int64,
            out_dim=(h_length + 1) // step,
        )

        # Create indices for W dimension (offset=2, step=3, length=60)
        w_offset = 2
        w_length = mrope_section[w_offset] * step
        w_indices = ops.range(
            w_offset,
            w_length,
            step,
            device=position_ids.device,
            dtype=DType.int64,
            out_dim=(w_length + 1) // 3,
        )

        # Extract values from H dimension at the specified head_dim positions
        # freqs[1] shape: (total_seq_len, head_dim // 2)
        # Gather along head_dim axis (axis=1) at positions h_indices
        # This selects those head_dim positions for ALL sequence positions
        total_seq_len = position_ids.shape[1]
        freqs_h_selected = ops.gather(
            input=freqs[1], indices=h_indices, axis=-1
        )  # (total_seq_len, num_h_indices)

        # Scatter these values back into freqs_t at the same head_dim positions
        # freqs_t shape: (total_seq_len, head_dim // 2)
        # We update positions h_indices in the head_dim dimension (axis=1) for all sequence positions
        h_indices_for_scatter = ops.tile(
            ops.unsqueeze(h_indices, 0), (total_seq_len, 1)
        )  # (total_seq_len, num_h_indices)
        freqs_t = ops.scatter(
            input=freqs_t,
            updates=freqs_h_selected,
            indices=h_indices_for_scatter,
            axis=1,
        )

        # Same for W dimension
        freqs_w_selected = ops.gather(
            input=freqs[2], indices=w_indices, axis=1
        )  # (total_seq_len, num_w_indices)
        w_indices_for_scatter = ops.tile(
            ops.unsqueeze(w_indices, 0), (total_seq_len, 1)
        )  # (total_seq_len, num_w_indices)
        freqs_t = ops.scatter(
            input=freqs_t,
            updates=freqs_w_selected,
            indices=w_indices_for_scatter,
            axis=1,
        )

        return freqs_t

    def freqs_cis_position_ids(self, position_ids: TensorValue) -> TensorValue:
        freqs = self.freqs_cis_base_position_ids(position_ids)
        d1, d2, d3 = freqs.shape  # (total_seq_len, head_dim // 2, 2)
        new_f_shape = [d1, d2 * d3]  # (total_seq_len, head_dim)
        self._freqs_cis = ops.reshape(freqs, new_f_shape)
        return self._freqs_cis
