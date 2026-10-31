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
"""Kimi K3's output head (ModuleV3), single-device or tensor-parallel.

Under TP each device holds ``vocab / n`` rows of ``lm_head``, and an
allgather joins the logits. At TP=1 the weight is not sharded and the
allgather is skipped.

Three details this mirrors exactly from V2's own docstring, easy to get wrong
by reaching for a framework default:

* ``lm_head`` is NOT tied to ``embed_tokens`` on this checkpoint
  (``config.tie_word_embeddings`` is False) even though the official
  reference lists it in ``_tied_weights_keys`` -- the tie must be decided
  from config, never assumed from the name list.
* the final norm is ``kimi_rms_norm`` (cast-then-multiply), not
  :class:`~max.nn.norm.RMSNorm`'s Gemma-style default -- and it is applied by
  the model-level residual mixture (``kimi_k3_attn_res_mix_norm``), whose
  output is this layer's input: the head itself receives an ALREADY-NORMED
  hidden state and only projects.
* only the last token's hidden state is projected for the ``LAST_TOKEN``
  case (the common path) -- the projection runs per-token, so gathering the
  last row first saves real arithmetic, not just a reorder. Norm-then-gather
  is per-row identical, which is what makes folding the norm into the
  mixture safe.

Scope cut, explicit: ``return_hidden_states`` (``ReturnHiddenStates``'s
``LAST``/``LAST_PER_DEVICE``/``ALL_NORMALIZED``/``ALL``/``SELECTED_LAYERS``
variants) is not ported -- this always returns ``ReturnHiddenStates.NONE``
behavior. V2's multi-device variants of that enum have no single-device
meaning to reduce to, and the capture-based variants are an advanced serving
feature this layer does not implement.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn.common_layers.mesh_axis import TP
from max.nn.legacy.transformer import ReturnLogits
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh, NamedMapping


def tensor_parallel_lm_head_weight(
    lm_head_weight: Tensor, mesh: DeviceMesh
) -> Tensor:
    """Shards ``lm_head_weight`` ``[vocab, hidden]`` on the vocabulary axis.

    ``kimi_k3_output_head`` allgathers the logits when the weight is sharded.
    """
    if mesh.axis_size(TP) == 1:
        return lm_head_weight
    return lm_head_weight.to(NamedMapping(mesh, (TP, None)))


def kimi_k3_output_head(
    hidden_states: Tensor,
    input_row_offsets: Tensor,
    return_n_logits: Tensor,
    lm_head_weight: Tensor,
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
) -> tuple[Tensor, ...]:
    """Projects the model-level stream's final norm to the vocabulary.

    Args:
        hidden_states: ``[total_seq_len, hidden]`` ragged stream, already
            normalized by the model-level residual mixture
            (``kimi_k3_attn_res_mix_norm``); the head only projects it.
        input_row_offsets: ``[batch_size + 1]`` ragged row offsets; the head
            gathers ``input_row_offsets[1:] - 1`` for the last-token case.
        return_n_logits: ``[1]`` per-sequence logit count, used by the
            VARIABLE branch.
        lm_head_weight: ``language_model.lm_head.weight``,
            ``[vocab, hidden]`` as stored in the checkpoint, hence the
            transpose in the projection. Sharded on the vocab axis under TP;
            see :func:`tensor_parallel_lm_head_weight`.
        return_logits: Which logits to return.

    Returns:
        ``(last_logits,)``, or ``(last_logits, logits, offsets)`` when
        ``return_logits`` is ``VARIABLE`` or ``ALL``.
    """
    mesh = lm_head_weight.mesh
    tp_sharded = TP in mesh.axis_names and mesh.axis_size(TP) > 1

    def project(h: Tensor) -> Tensor:
        logits = h @ lm_head_weight.T
        if tp_sharded:
            logits = F.allgather(logits, tensor_axis=-1, mesh_axis=TP)
        return F.cast(logits, DType.float32)

    last_token_indices = input_row_offsets[1:] - 1
    last_token_h = F.gather(hidden_states, last_token_indices, axis=0)
    last_logits = project(last_token_h)

    if return_logits == ReturnLogits.VARIABLE:
        n = return_n_logits[0]
        # One sequence's own last `n` rows: `row_offsets[1:]` (exclusive end)
        # minus `n, n-1, ..., 1`, mirroring V2's `range(n, 0, -1)` offset.
        offsets_end = F.unsqueeze(input_row_offsets[1:], -1)
        step_back = F.arange(n, 0, -1, dtype=offsets_end.dtype)
        indices = F.reshape(offsets_end - step_back, [-1])
        logits = project(F.gather(hidden_states, indices, axis=0))
        offsets = F.arange(
            0, indices.shape[0] + n, n, dtype=input_row_offsets.dtype
        )
        return last_logits, logits, offsets
    if return_logits == ReturnLogits.ALL:
        return last_logits, project(hidden_states), input_row_offsets

    return (last_logits,)
