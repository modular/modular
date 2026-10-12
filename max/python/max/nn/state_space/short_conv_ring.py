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
"""Depthwise causal short convolution over a ring of past inputs.

The conv state is ``[slots, ring_len, channels]``, the input at position
``p`` at ring index ``p % ring_len``. :func:`short_conv_ring` computes
``x + conv(x)`` with pre-chunk taps read from the ring and, with ``commit``,
writes each sequence's last ``ring_len`` inputs in the same launch. One op
covers decode, prefill, mixed batches and speculative verify. ``ring_len`` is
``kernel_size - 1`` plus the largest rollback a verify step can cause.
"""

from __future__ import annotations

from max.graph import BufferValue, TensorType, TensorValue, ops


def short_conv_ring(
    x: TensorValue,
    weight: TensorValue,
    ring: BufferValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    conv_rows: TensorValue,
    layer_row: TensorValue,
    *,
    commit: bool,
) -> TensorValue:
    """Returns ``x + conv(x)`` over a ragged batch, optionally committing the
    chunk to ``ring``.

    Args:
        x: ``[total_seq_len, channels]`` input.
        weight: ``[channels, kernel_size]`` taps; the last multiplies the
            current token.
        ring: ``[slots, ring_len, channels]`` conv state; written in place
            when ``commit`` is set.
        input_row_offsets: ``[batch + 1]`` uint32.
        positions: ``[total_seq_len]`` uint32 position per token.
        conv_rows: ``[num_layers, batch]`` uint32 ring slot per sequence.
        layer_row: Scalar uint32 CPU row of ``conv_rows`` this layer reads.
        commit: Whether to write each sequence's last ``ring_len`` rows of
            ``x`` into its slot of ``ring``.

    Returns:
        Same shape and dtype as ``x``.
    """
    return ops.inplace_custom(
        "mo.short_conv_ring",
        device=x.device,
        values=[
            x,
            weight,
            ring,
            input_row_offsets,
            positions,
            conv_rows,
            layer_row,
        ],
        out_types=[TensorType(x.dtype, x.shape, device=x.device)],
        parameters={"commit": commit},
    )[0].tensor
