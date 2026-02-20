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
"""Fused RoPE kernel for vision models."""

from __future__ import annotations

from max.graph import DeviceRef, TensorType, TensorValue, TensorValueLike, ops


def fused_qk_rope_vision(
    query: TensorValueLike,
    key: TensorValueLike,
    cos: TensorValueLike,
    sin: TensorValueLike,
    repeat_interleave: bool = True,
) -> tuple[TensorValue, TensorValue]:
    """Applies RoPE to query and key tensors using a fused kernel.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim].
        key: Key tensor [batch, seq_len, num_k_heads, head_dim].
        cos: Cosine frequency tensor [seq_len, head_dim].
        sin: Sine frequency tensor [seq_len, head_dim].
        repeat_interleave: Whether frequencies are repeated (Flux2/LLama3) or
            interleaved. Flux2 uses repeat-interleave (True).

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    # Convert TensorValueLike to TensorValue if needed
    query_val = TensorValue(query)
    key_val = TensorValue(key)
    cos_val = TensorValue(cos)
    sin_val = TensorValue(sin)

    device = DeviceRef.from_device(query_val.device)

    # Define output types (same shapes/dtypes as inputs)
    out_types = [
        TensorType(
            dtype=query_val.dtype,
            shape=query_val.shape,
            device=device,
        ),
        TensorType(
            dtype=key_val.dtype,
            shape=key_val.shape,
            device=device,
        ),
    ]

    # Call the registered internal op
    # The kernel signature is: (q_out, k_out, query, key, freqs_cos, freqs_sin, ctx)
    results = ops.custom(
        name="mo.fused_qk_rope_vision",
        device=device,
        values=[query_val, key_val, cos_val, sin_val],
        out_types=out_types,
    )

    return results[0].tensor, results[1].tensor
