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
"""MiMo-V2.6-Flash pipeline model."""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import MultiKVCacheParams, spec_decode_cache_slack
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gpt_oss.batch_processor import (
    GptOssBatchProcessor,
)
from max.pipelines.architectures.gpt_oss.model import GptOssInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    BatchProcessor,
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.weights import HuggingFaceRepo

from .mimo_v2 import MiMoV2
from .model_config import MiMoV2Config

_WEIGHT_INDEX = "model.safetensors.index.json"


def text_model_weights(
    repo: HuggingFaceRepo, weights: Weights
) -> dict[str, Weights]:
    """Returns the weights the text model's safetensors index lists.

    The repo also ships the audio tokenizer and the DFlash drafter as
    safetensors in subdirectories, which weight discovery globs in. The text
    model's tensors are those of the index in the model's folder (the repo's
    ``subfolder``, or its root); a checkpoint without an index is a single
    file, all of it the model's.

    Args:
        repo: The model's repo.
        weights: Every tensor weight discovery found.

    Returns:
        The text model's tensors, by checkpoint name.
    """
    index = (
        f"{repo.subfolder}/{_WEIGHT_INDEX}" if repo.subfolder else _WEIGHT_INDEX
    )
    if repo.repo_type == "local":
        path = os.path.join(repo.local_path, index)
        if not os.path.isfile(path):
            return dict(weights.items())
    else:
        try:
            path = huggingface_hub.hf_hub_download(
                repo.repo_id, index, revision=repo.revision
            )
        except EntryNotFoundError:
            return dict(weights.items())
    with open(path) as f:
        names = set(json.load(f)["weight_map"])
    return {name: w for name, w in weights.items() if name in names}


class MiMoV2Model(
    AlwaysSignalBuffersMixin, GraphPipelineModelWithKVCache[TextContext]
):
    """Pipeline model for MiMo-V2.6-Flash.

    The graph takes GPT OSS's inputs (tokens, per-device row offsets, signal
    buffers and a ``{sliding_attention, full_attention}`` KV tree), so GPT
    OSS's inputs and batching are reused.
    """

    model_config_cls: ClassVar[type[Any]] = MiMoV2Config
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        GptOssBatchProcessor
    )
    _strict_state_dict_loading = True

    _language_weights_dict: dict[str, Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled and initialized model."""
        return self._models[GraphRole.DECODER]

    def _load_state_dict(self) -> dict[str, Any]:
        """Adapts the text model's tensors, :func:`text_model_weights`.

        The adapter must see exactly those to prove every one is consumed or
        ignored.
        """
        self._language_weights_dict = _adapted_state_dict(
            self.pipeline_config, self.weights, self.adapter
        )
        return self._language_weights_dict

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[MiMoV2, MiMoV2Config]:
        """Builds the MiMo-V2 module, sharded over the config's devices.

        A caller that sized the cache passes its KV params, and the RoPE
        table then reaches the positions a speculative step writes past the
        last committed one.

        Raises:
            ValueError: If the selected layers aren't in layer order, the
                order the module captures them in.
        """
        layers = list(context.target_layer_ids)
        if layers != sorted(layers):
            raise ValueError(
                "MiMo-V2 captures its layers in layer order; got"
                f" target_layer_ids {layers}."
            )
        max_seq_len = context.max_seq_len
        kv_params = context.kv_params
        if kv_params is not None:
            assert isinstance(kv_params, MultiKVCacheParams)
            max_seq_len += spec_decode_cache_slack(kv_params)
        config = MiMoV2Config.initialize(
            context.pipeline_config, max_seq_len=max_seq_len
        )
        if kv_params is not None:
            config.kv_params = kv_params
        config.return_logits = context.return_logits
        config.return_hidden_states = context.return_hidden_states
        config.target_layer_ids = layers or None
        module = MiMoV2(config)
        module.load_state_dict(
            _adapted_state_dict(
                context.pipeline_config, context.weights, context.adapter
            ),
            weight_alignment=1,
            strict=cls._strict_state_dict_loading,
        )
        return module, config

    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        del state_dict
        config = MiMoV2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.return_logits = self.return_logits
        return config

    def graph_specs(self, model_config: MiMoV2Config) -> dict[str, GraphSpec]:
        """Returns the decoder graph."""
        nn_model = MiMoV2(model_config)
        nn_model.load_state_dict(
            self._language_weights_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        # The graph's weights registry holds what it binds; drop the rest.
        self._language_weights_dict = {}
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "mimo_v2",
                self._bind_graph_inputs(nn_model.input_tree()),
                weights=nn_model.state_dict(auto_initialize=False),
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the model on inputs from :class:`GptOssBatchProcessor`."""
        assert isinstance(model_inputs, GptOssInputs)
        assert model_inputs.kv_cache_inputs is not None
        assert isinstance(model_inputs.input_row_offsets, list)
        outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *model_inputs.input_row_offsets,
            *model_inputs.signal_buffers,
            *tree.leaves(model_inputs.kv_cache_inputs),
        )
        if len(outputs) == 3:
            next_token_logits, logits, offsets = outputs
            assert isinstance(next_token_logits, Buffer)
            assert isinstance(logits, Buffer)
            assert isinstance(offsets, Buffer)
            return ModelOutputs(
                logits=logits,
                next_token_logits=next_token_logits,
                logit_offsets=offsets,
            )
        logits = outputs[0]
        assert isinstance(logits, Buffer)
        return ModelOutputs(logits=logits, next_token_logits=logits)


def _adapted_state_dict(
    pipeline_config: PipelineConfig,
    weights: Weights,
    adapter: WeightsAdapter | None,
) -> dict[str, Any]:
    assert adapter is not None
    return adapter(
        text_model_weights(
            pipeline_config.model.huggingface_weight_repo, weights
        ),
        huggingface_config=pipeline_config.model.huggingface_config,
        pipeline_config=pipeline_config,
    )
