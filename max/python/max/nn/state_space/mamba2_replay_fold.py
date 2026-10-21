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
"""Python wrapper for the Mamba-2 speculative replay fold."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferValue, TensorValue, ops


def mamba2_replay_fold(
    conv_pool: BufferValue,
    ssm_pool: BufferValue,
    conv_rows: TensorValue,
    ssm_rows: TensorValue,
    replay_bounds: TensorValue,
    projected: Sequence[TensorValue],
    convolved: Sequence[TensorValue],
    A: Sequence[TensorValue],
    dt_bias: Sequence[TensorValue],
) -> None:
    """Commits each request's accepted verify rows to every Mamba-2 layer.

    For each layer and request, the conv window becomes the last
    ``width - 1`` accepted conv inputs and the SSM state folds in the
    accepted rows, all in one launch. The SSM state lands exactly where
    replaying those rows through
    :func:`~max.nn.state_space.mamba2_ssd_chunk_scan_varlen_fwd_inplace` would
    put it. Every request continues its live state. GPU only.

    Args:
        conv_pool: The ``[max_slots, conv_dim, width - 1]`` conv windows,
            updated in place.
        ssm_pool: The ``[max_slots, nheads, head_dim, dstate]`` SSM states,
            updated in place.
        conv_rows: The ``[num_layers, batch]`` uint32 conv-window rows.
        ssm_rows: The ``[num_layers, batch]`` uint32 SSM-state rows.
        replay_bounds: The ``[2 * batch]`` int32 accepted ``(start, end)``
            rows of each request.
        projected: Each layer's ``in_proj`` output rows,
            ``[rows, nheads * head_dim + conv_dim + nheads]``: the gate, the
            conv input and ``dt``.
        convolved: Each layer's conv output rows, ``[rows, conv_dim]``: the
            scan's ``x``, ``B`` and ``C``.
        A: Each layer's ``[nheads]`` decay, already ``-exp(A_log)``.
        dt_bias: Each layer's ``[nheads]`` ``dt`` bias.
    """
    num_layers = len(projected)
    if not (num_layers == len(convolved) == len(A) == len(dt_bias)):
        raise ValueError(
            "mamba2_replay_fold needs the same number of projected, convolved,"
            f" A and dt_bias tensors; got {len(projected)}, {len(convolved)},"
            f" {len(A)} and {len(dt_bias)}"
        )
    # The op takes one variadic group of rank-2 tensors, in four runs.
    heads = [ops.unsqueeze(t, 0) for t in (*A, *dt_bias)]
    ops.inplace_custom(
        "mamba2_replay_fold",
        conv_pool.device,
        [
            conv_pool,
            ssm_pool,
            conv_rows,
            ssm_rows,
            replay_bounds,
            *projected,
            *convolved,
            *heads,
        ],
        parameters={"dt_softplus": True},
    )
