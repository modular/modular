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
"""Config for EAGLE Llama3 draft models."""

from __future__ import annotations

from collections.abc import Mapping

from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.legacy.layer import Module
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.registration_draft import DraftRegistration
from max.pipelines.speculative import SpeculativeConfig
from max.pipelines.speculative.eagle import EagleDraftAdapter
from transformers import AutoConfig

from ..llama3.model_config import Llama3Config

__all__ = ["EagleLlama3Config", "Llama3Config", "eagle_draft_registration"]


class EagleLlama3Config(Llama3Config):
    """Checkpoint configuration of the registered EAGLE draft component."""


def configure_eagle_draft(
    context: ModelLoadContext, state_dict: Mapping[str, WeightData]
) -> EagleLlama3Config:
    """Finalizes the draft's own config against its checkpoint and planned KV."""
    pipeline = context.pipeline_config
    checkpoint = pipeline.draft_model or pipeline.model
    config = EagleLlama3Config.initialize(
        pipeline, checkpoint, max_seq_len=context.max_seq_len
    )
    assert isinstance(context.kv_params, KVCacheParams)
    config.kv_params = context.kv_params
    config.devices = list(context.kv_params.devices)
    config.finalize(
        huggingface_config=checkpoint.huggingface_config,
        state_dict=dict(state_dict),
        norm_method="rms_norm",
        attention_bias=False,
        return_logits=ReturnLogits.LAST_TOKEN,
        return_hidden_states=ReturnHiddenStates.LAST,
    )
    return config


def eagle_proposer(
    module: Module,
    config: EagleLlama3Config,
    *,
    speculative: SpeculativeConfig,
    takes_distributed_inputs: bool,
    target_text_config: AutoConfig,
) -> EagleDraftAdapter:
    """Binds the registered computation to EAGLE's existing sequential driver.

    Raises:
        ValueError: If the target takes the distributed signature, which
            only an MLA EAGLE3 head runs beside.
    """
    del speculative, target_text_config
    if takes_distributed_inputs:
        raise ValueError(
            "A Llama EAGLE draft doesn't run beside a distributed target."
        )
    return EagleDraftAdapter(module, config.hidden_size)


eagle_draft_registration = DraftRegistration(
    method="eagle",
    configure=configure_eagle_draft,
    create_proposer=eagle_proposer,
    shared_modules=("embed_tokens", "lm_head"),
    optional_weights=frozenset({"fc.bias"}),
)
