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

"""Implements the DeepSeek-V4-Flash PipelineModel."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    RaggableContext,
    SingleReplicaRaggedBatchProcessor,
    build_single_replica_ragged_token_arrays,
    ragged_kv_symbolic_inputs,
    single_replica_context_batch,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights

from .deepseekV4 import DeepseekV4
from .layers.quantization import fp8_block_quant_config
from .model_config import DeepseekV4Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV4Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4 model execution."""

    tokens: Buffer
    """Input token IDs, ragged."""

    input_row_offsets: Buffer
    """Offsets of each sequence in the ragged ``tokens``."""

    return_n_logits: Buffer
    """Number of trailing logits to return."""

    signal_buffers: list[Buffer] = field(default_factory=list)
    """One per device when there is more than one, else empty."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
        )


class DeepseekV4BatchProcessor(
    SingleReplicaRaggedBatchProcessor[
        TextContext, DeepseekV4Inputs, DeepseekV4Config
    ]
):
    """Ragged single-replica batching; the stock ragged KV input order."""

    _include_signal_buffers: ClassVar[bool] = True

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        # A single device keeps the signal-free signature, and with it the
        # compiled graph, of the single-device bringup.
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=(
                self._include_signal_buffers and len(device_refs) > 1
            ),
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV4Inputs:
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        device0 = self.runtime.devices[0]
        return DeepseekV4Inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            signal_buffers=list(self.runtime.signal_buffers),
        )


def set_checkpoint_dtypes(
    config: DeepseekV4Config,
    huggingface_config: Any,
    *,
    norm_dtype: DType,
    embed_dtype: DType,
) -> None:
    """Sets what the config reads off the adapted checkpoint.

    Args:
        config: The config to finish.
        huggingface_config: The checkpoint's HuggingFace config.
        norm_dtype: The adapted dtype of the checkpoint's norm weights.
        embed_dtype: The stored dtype of the checkpoint's bf16 weights,
            ``embed`` among them.

    Raises:
        ValueError: If the encoding is fp8 but the checkpoint declares no fp8
            ``quantization_config``.
    """
    config.norm_dtype = norm_dtype
    if config.quantization_encoding == "float8_e4m3fn":
        # The encoding names the fp8 projections only. The activation dtype
        # is the reference's default dtype, bf16, which is also what the
        # checkpoint stores ``embed`` / ``ffn.gate`` /
        # ``indexer.weights_proj`` in and the reference declares them as;
        # those follow ``config.dtype``, and so does ``head`` (bf16 GEMM, see
        # ``head_logits``). The other weights the reference declares float32
        # (norms, compressor ``wkv`` / ``wgate``) are declared so by the
        # modules and upcast by the adapter.
        quant_config = fp8_block_quant_config(
            getattr(huggingface_config, "quantization_config", None),
            config.num_hidden_layers,
        )
        if quant_config is None:
            raise ValueError(
                "quantization_encoding float8_e4m3fn needs a checkpoint "
                "that declares its fp8 quantization_config"
            )
        config.quant_config = quant_config
        config.dtype = embed_dtype


class DeepseekV4Model(GraphPipelineModelWithKVCache[TextContext]):
    """A DeepSeek-V4-Flash pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV4Config
    # As wide as the base's, so a subclass can bring its own processor.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = (
        DeepseekV4BatchProcessor
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
        """The compiled and initialized MAX Engine model."""
        return self._models[GraphRole.DECODER]

    def _load_state_dict(self) -> dict[str, Any]:
        self._language_weights_dict = super()._load_state_dict()
        return self._language_weights_dict

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[DeepseekV4, DeepseekV4Config]:
        """Builds the trunk, without the DSpark stages its adapter drops.

        A caller that sized the cache passes its KV params. Under
        ``SELECTED_LAYERS`` the trunk returns the state DSpark reads, which
        is collected at the checkpoint's ``dspark_target_layer_ids``.

        Raises:
            ValueError: If ``target_layer_ids`` names other layers.
        """
        config = DeepseekV4Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        layers = list(context.target_layer_ids)
        if layers and layers != config.dspark_target_layer_ids:
            raise ValueError(
                "DeepSeek-V4 collects the states at its"
                f" dspark_target_layer_ids {config.dspark_target_layer_ids};"
                f" got target_layer_ids {layers}."
            )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, KVCacheParamInterface)
            config.kv_params = context.kv_params
        config.return_logits = context.return_logits
        config.return_hidden_states = context.return_hidden_states
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        set_checkpoint_dtypes(
            config,
            context.huggingface_config,
            norm_dtype=state_dict["norm.weight"].dtype,
            embed_dtype=state_dict["embed.weight"].dtype,
        )
        module = DeepseekV4(config)
        module.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=cls._strict_state_dict_loading,
        )
        return module, config

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV4Config:
        model_config = self.arch_config_as(DeepseekV4Config)
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states
        set_checkpoint_dtypes(
            model_config,
            self.huggingface_config,
            norm_dtype=state_dict["norm.weight"].dtype,
            embed_dtype=state_dict["embed.weight"].dtype,
        )
        return model_config

    def graph_specs(
        self, model_config: DeepseekV4Config
    ) -> dict[str, GraphSpec]:
        """Returns the decoder graph, tensor parallel on several devices."""
        nn_model = DeepseekV4(model_config)
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
                "deepseekV4",
                self._bind_graph_inputs(nn_model.input_tree()),
                # Taken before the trace shards the weights.
                weights=nn_model.state_dict(auto_initialize=False),
            )
        }

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(DeepseekV4Inputs, model_inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)
