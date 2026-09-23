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
``p`` at ring index ``p % ring_len``. :func:`short_conv_ring_fwd` computes
``x + conv(x)`` with pre-chunk taps read from the ring and writes nothing;
:func:`short_conv_ring_commit` then writes each sequence's last ``ring_len``
inputs. One pair of ops covers decode, prefill, mixed batches and
speculative verify. ``ring_len`` is ``kernel_size - 1`` plus the largest
rollback a verify step can cause.
"""

from __future__ import annotations

from max.graph import BufferValue, TensorType, TensorValue, ops


def short_conv_ring_fwd(
    x: TensorValue,
    weight: TensorValue,
    ring: BufferValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    conv_row: TensorValue,
) -> TensorValue:
    """Returns ``x + conv(x)`` over a ragged batch; reads ``ring``, writes
    nothing.

    Args:
        x: ``[total_seq_len, channels]`` input.
        weight: ``[channels, kernel_size]`` taps; the last multiplies the
            current token.
        ring: ``[slots, ring_len, channels]`` conv state.
        input_row_offsets: ``[batch + 1]`` uint32.
        positions: ``[total_seq_len]`` uint32 position per token.
        conv_row: ``[batch]`` uint32 ring slot per sequence.

    Returns:
        Same shape and dtype as ``x``.
    """
    return ops.inplace_custom(
        "mo.short_conv_ring_fwd",
        device=x.device,
        values=[x, weight, ring, input_row_offsets, positions, conv_row],
        out_types=[TensorType(x.dtype, x.shape, device=x.device)],
    )[0].tensor


def short_conv_ring_commit(
    x: TensorValue,
    ring: BufferValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    conv_row: TensorValue,
) -> None:
    """Writes each sequence's last ``ring_len`` rows of ``x`` into its slot
    of ``ring``. Issue after every reader of ``ring`` in the same forward.

    Args:
        x: ``[total_seq_len, channels]`` conv input.
        ring: ``[slots, ring_len, channels]`` conv state, written in place.
        input_row_offsets: ``[batch + 1]`` uint32.
        positions: ``[total_seq_len]`` uint32 position per token.
        conv_row: ``[batch]`` uint32 ring slot per sequence.
    """
    ops.inplace_custom(
        "mo.short_conv_ring_commit",
        device=x.device,
        values=[ring, x, input_row_offsets, positions, conv_row],
        out_types=[],
    )
