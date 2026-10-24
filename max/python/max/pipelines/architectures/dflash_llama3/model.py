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
"""The arch config of the DFlash draft architecture.

The DFlash draft (HuggingFace architecture ``DFlashDraftModel``) runs only
beside a target, so it has no pipeline model: its arch config builds the
draft's module for a unified spec-decode graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative import validate_draft_state_dict
from max.pipelines.speculative._dflash import (
    DflashDraftHFConfig,
    parse_dflash_draft_hf_config,
)

from ..llama3.model_config import Llama3Config
from .dflash_llama3 import DFlashLlama3


@dataclass(frozen=True)
class DFlashLlama3Config:
    """The draft's Llama 3 config and the DFlash fields its checkpoint names."""

    llama: Llama3Config
    dflash: DflashDraftHFConfig

    @property
    def hidden_size(self) -> int:
        """Returns the draft's hidden size."""
        return self.llama.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the draft's vocabulary size."""
        return self.llama.vocab_size

    @property
    def mask_token_id(self) -> int:
        """Returns the token that fills the block's drafted slots."""
        return self.dflash.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the draft reads."""
        return self.dflash.target_layer_ids


class DFlashLlama3ArchConfig(Llama3Config):
    """The DFlash draft's Llama 3 config, which builds the draft's module."""

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DFlashLlama3, DFlashLlama3Config]:
        """Builds the DFlash draft from the draft checkpoint ``context`` names.

        The draft computes in its checkpoint's own precision and has no
        embedding or head; it embeds and projects through its target's.

        Raises:
            ValueError: If the method isn't ``dflash``.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        assert draft_model.max_length is not None
        if speculative.speculative_method != "dflash":
            raise ValueError(
                "The DFlashDraftModel draft is a DFlash draft and requires"
                " --speculative-method dflash, got"
                f" {speculative.speculative_method!r}."
            )
        hf_config = draft_model.huggingface_config
        state_dict = parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            context.adapter,
            hf_config=hf_config,
            model_config=draft_model,
        )
        llama = Llama3Config.initialize_from_config(
            pipeline_config,
            hf_config,
            draft_model,
            max_seq_len=draft_model.max_length,
        )
        llama.sliding_window = getattr(hf_config, "sliding_window", None)
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParams)
            llama.kv_params = context.kv_params
            llama.devices = list(context.kv_params.devices)
        llama.finalize(
            huggingface_config=hf_config,
            state_dict=state_dict,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.LAST,
        )
        config = DFlashLlama3Config(
            llama, parse_dflash_draft_hf_config(hf_config)
        )
        layer_types = list(getattr(hf_config, "layer_types", None) or [])
        draft = DFlashLlama3(
            llama,
            num_context_features=len(config.target_layer_ids),
            layer_types=layer_types or None,
        )
        draft.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        validate_draft_state_dict(
            draft.raw_state_dict().keys(), state_dict.keys()
        )
        return draft, config
