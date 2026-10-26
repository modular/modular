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
"""The ModuleV3 Gemma 4 language model's call.

Only the MTP assistant drafts for it. The assistant cross-attends into the
target's sliding-window cache on every device the target spans.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import (
    SequentialBatch,
    tensor_compat_verify,
)
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified

from .gemma4 import Gemma4TextModel

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

_SLIDING_KV = "sliding_attention"

_FULL_KV = "full_attention"


def verify(
    target: Any,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a ModuleV3 Gemma 4 language model.

    Raises:
        ValueError: If the target wasn't built to return the last normalized
            layer the MTP assistant reads.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.ALL_NORMALIZED:
        raise ValueError(
            "A ModuleV3 Gemma 4 target verifies only for a draft that reads"
            f" its last normalized layer. Got {hidden_states.value}."
        )
    assert isinstance(target, Gemma4TextModel)
    on_tensors = tensor_compat_verify(partial(_verify, target))

    def on_batch(batch: SequentialBatch | BlockBatch) -> Verified[Any]:
        # Only the MTP assistant drafts for this target, a token a step.
        assert isinstance(batch, SequentialBatch)
        return on_tensors(batch)

    return on_batch


def _verify(
    target: Gemma4TextModel, batch: SequentialBatch[Tensor, Tensor, Tensor]
) -> Verified[list[Tensor], Tensor]:
    replicated = DeviceMapping.replicated(target.mesh)
    # Images arrive only in prefill, which merges no drafts into a row.
    outputs = target(
        batch.merged_tokens,
        PagedCacheValues.from_upstream(batch.kv_collections, replicated),
        PagedCacheValues.from_upstream(
            batch.passthrough_kv[_FULL_KV], replicated
        ),
        batch.return_n_logits,
        # The target broadcasts the merged tokens and offsets itself.
        batch.merged_offsets,
        batch.vision_embeddings[0],
        batch.vision_scatter_indices[0],
    )
    # (last_logits, logits, offsets, hs_0..hs_{n-1}). The logits are
    # replicated after the vocab allgather, so the first device's serves.
    return Verified(
        logits=outputs[1].local_shards[0],
        hidden=list(outputs[3 : 3 + len(target.mesh.devices)]),
    )


def _check(speculative: SpeculativeConfig, structured_output: bool) -> None:
    """Refuses every method but MTP."""
    del structured_output
    if not speculative.is_mtp():
        raise ValueError(
            "A ModuleV3 Gemma 4 target speculates only with its MTP"
            f" assistant, got --speculative-method"
            f" {speculative.speculative_method!r}."
        )


ADAPTER = TargetAdapter(primary_kv=_SLIDING_KV, verify=verify, check=_check)
