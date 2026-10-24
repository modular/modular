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
"""MiMo-V2's call, on every device the target is sharded over."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from max.graph import TensorValue
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified
from max.pipelines.speculative.unified_graph_ops import mask_padded_tail

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

PRIMARY_KV = "full_attention"

_SLIDING_KV = "sliding_attention"


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a MiMo-V2 target, returning its layer taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the head's padding rows, past the tokenizer the draft config
    records, at negative infinity.

    Raises:
        ValueError: If the target wasn't built to return selected layers,
            which only block drafts read.
    """
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A MiMo-V2 target verifies only for block drafts, which read"
            f" selected layers. Got {hidden_states.value}."
        )
    return partial(
        _verify,
        target,
        target_config.vocab_size,
        draft_config.sampleable_vocab_size,
    )


def _verify(
    target: Module,
    vocab_size: int,
    sampleable_vocab_size: int,
    batch: SequentialBatch | BlockBatch,
) -> Verified[list[TensorValue]]:
    assert isinstance(batch, BlockBatch)
    outputs = target(
        tokens=batch.merged_tokens,
        signal_buffers=batch.signal_buffers,
        sliding_kv_collections=batch.passthrough_kv[_SLIDING_KV],
        full_kv_collections=batch.kv_collections,
        return_n_logits=batch.return_n_logits,
        input_row_offsets=list(batch.merged_offsets_per_dev),
    )
    return Verified(
        logits=mask_padded_tail(
            outputs.logits["logits"], vocab_size, sampleable_vocab_size
        ),
        hidden=fuse_captured_hidden_states(
            captures_by_device(
                list(outputs.hidden_states.tensors()), batch.n_devs
            )
        ),
    )


ADAPTER = TargetAdapter(primary_kv=PRIMARY_KV, verify=verify)
