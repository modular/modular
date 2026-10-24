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

from __future__ import annotations

from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.speculative import DraftAliases, validate_draft_state_dict

from .eagle_llama3 import EagleLlama3
from .model_config import Llama3Config


class EagleLlama3ArchConfig(Llama3Config):
    """The EAGLE draft's Llama 3 config, which builds the draft's module.

    The draft runs only beside a target, so it has no pipeline model.
    """

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[EagleLlama3, Llama3Config]:
        """Builds the EAGLE draft from the draft checkpoint ``context`` names.

        The draft computes in its checkpoint's own precision. It embeds and
        projects through its target's embedding and head, so spec decoding
        shares those onto it.

        Raises:
            ValueError: If the method isn't ``eagle``.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_eagle():
            raise ValueError(
                "The LlamaForCausalLMEagle draft is an EAGLE draft and"
                " requires --speculative-method eagle, got"
                f" {speculative.speculative_method!r}."
            )
        draft_hf_config = draft_model.huggingface_config
        state_dict = parse_state_dict_from_weights(
            pipeline_config,
            context.weights,
            context.adapter,
            hf_config=draft_hf_config,
            model_config=draft_model,
        )
        config = Llama3Config.initialize_from_config(
            pipeline_config,
            draft_hf_config,
            draft_model,
            max_seq_len=context.max_seq_len,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParams)
            config.kv_params = context.kv_params
            config.devices = list(context.kv_params.devices)
        # The EAGLE proposer switches the draft's modes around prefill.
        config.finalize(
            huggingface_config=draft_hf_config,
            state_dict=state_dict,
            norm_method="rms_norm",
            attention_bias=False,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.LAST,
        )
        draft = EagleLlama3(config)
        draft.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            # The draft replaces its norms, and EAGLE-1 checkpoints trained
            # without an fc bias ship none.
            strict=False,
        )
        validate_draft_state_dict(
            (name for name in draft.raw_state_dict() if name != "fc.bias"),
            state_dict.keys(),
            DraftAliases(always=("embed_tokens.", "lm_head.")),
        )
        return draft, config
