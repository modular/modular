# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements the DeepseekV2 nn.model."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
from max.driver import Device, DeviceSpec, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import LogProbabilities, TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    compute_log_probabilities,
    upper_bounded_default,
)
from transformers import AutoConfig

from .deepseekV2 import DeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV2Inputs(ModelInputs):
    """A class representing inputs for the DeepseekV2 model.

    This class encapsulates the input tensors required for the DeepseekV2 model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of logits to return
    """

    tokens: Tensor
    input_row_offsets: Tensor
    return_n_logits: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: Tensor | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.kv_cache_inputs = kv_cache_inputs
        if return_n_logits is None:
            # Provide a default value if none is provided
            self.return_n_logits = Tensor.from_numpy(
                np.array([1], dtype=np.int64)
            ).to(tokens.device)
        else:
            self.return_n_logits = return_n_logits


class DeepseekV2Model(PipelineModel[TextContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_n_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        if pipeline_config.model_config.device_specs[0] == DeviceSpec.cpu():
            msg = "DeepseekV2 currently only supported on gpu."
            raise ValueError(msg)

        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_n_logits,
        )
        self.return_n_logits = return_n_logits
        self.model = self.load_model(session)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(DeepseekV2Inputs, model_inputs)
        # keep mypy happy.
        assert model_inputs.kv_cache_inputs is not None, (
            "DeepseekV2 has KV cache inputs"
        )
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[1]),
                logit_offsets=cast(Tensor, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV2Inputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return DeepseekV2Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ).to(self.devices[0]),
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV2Inputs:
        prev_model_inputs = cast(DeepseekV2Inputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        return DeepseekV2Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return DeepseekV2Config.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for DeepseekV2, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=DeepseekV2Config.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.huggingface_config.num_hidden_layers,
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=DeepseekV2Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=huggingface_config.num_hidden_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # Read in weights.
        if not isinstance(self.weights, SafetensorWeights):
            msg = "only safetensors weights supported in DeepseekV2."
            raise ValueError(msg)

        logger.info("Building and compiling model...")
        before = time.perf_counter()

        pipeline_config = self.pipeline_config
        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        kv_params = DeepseekV2Config.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]
        model_config = DeepseekV2Config(
            attention_bias=huggingface_config.attention_bias,
            attention_dropout=huggingface_config.attention_dropout,
            aux_loss_alpha=huggingface_config.aux_loss_alpha,
            bos_token_id=huggingface_config.bos_token_id,
            eos_token_id=huggingface_config.eos_token_id,
            first_k_dense_replace=huggingface_config.first_k_dense_replace,
            hidden_act=huggingface_config.hidden_act,
            hidden_size=huggingface_config.hidden_size,
            initializer_range=huggingface_config.initializer_range,
            intermediate_size=huggingface_config.intermediate_size,
            kv_lora_rank=huggingface_config.kv_lora_rank,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            moe_intermediate_size=huggingface_config.moe_intermediate_size,
            moe_layer_freq=huggingface_config.moe_layer_freq,
            n_group=huggingface_config.n_group,
            n_routed_experts=huggingface_config.n_routed_experts,
            n_shared_experts=huggingface_config.n_shared_experts,
            norm_topk_prob=huggingface_config.norm_topk_prob,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_experts_per_tok=huggingface_config.num_experts_per_tok,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            pretraining_tp=huggingface_config.pretraining_tp,
            q_lora_rank=huggingface_config.q_lora_rank,
            qk_nope_head_dim=huggingface_config.qk_nope_head_dim,
            qk_rope_head_dim=huggingface_config.qk_rope_head_dim,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_scaling=huggingface_config.rope_scaling,
            rope_theta=huggingface_config.rope_theta,
            routed_scaling_factor=huggingface_config.routed_scaling_factor,
            scoring_func=huggingface_config.scoring_func,
            seq_aux=huggingface_config.seq_aux,
            tie_word_embeddings=huggingface_config.tie_word_embeddings,
            topk_group=huggingface_config.topk_group,
            topk_method=huggingface_config.topk_method,
            use_cache=huggingface_config.use_cache,
            v_head_dim=huggingface_config.v_head_dim,
            vocab_size=huggingface_config.vocab_size,
            dtype=self.dtype,
            kv_params=kv_params,
            devices=device_refs,
        )
        nn_model = DeepseekV2(model_config)

        # print("Expected weights:")
        # print("\n".join(f"{k}: {v}\n" for k, v in nn_model.raw_state_dict().items()))

        nn_model.load_state_dict(state_dict, weight_alignment=1)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU()
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.GPU(),
        )
        return_n_logits_type = TensorType(
            DType.int64,
            shape=["return_n_logits"],
            device=DeviceRef.GPU(),
        )
        kv_cache_types = self.kv_manager.input_symbols()[0]
        with Graph(
            "DeepseekV2",
            input_types=[
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *kv_cache_types,
            ],
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            # This is just needed for type checking.
            kv_cache_tensors = [v.tensor for v in kv_cache_inputs]
            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets,
                kv_cache_inputs=kv_cache_tensors,
                return_n_logits=return_n_logits.tensor,
            )
            graph.output(*outputs)

        model = session.load(graph, weights_registry=nn_model.state_dict())
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return model

    def compute_log_probabilities(
        self,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        if any(echo for echo in batch_echo):
            if model_outputs.logits is None:
                warnings.warn(
                    "Could not get logprobs with echo because the full logits"
                    f" were not returned by {self.pipeline_config.model_config.model_path}"
                    " model. Please ensure that this model is started with "
                    "`--enable-echo`."
                )
                assert not self.pipeline_config.enable_echo, (
                    "Echo was enabled but logits were not returned."
                )
                return None
            logits = model_outputs.logits.to_numpy()
        assert model_outputs.next_token_logits
        next_token_logits = model_outputs.next_token_logits.to_numpy()

        sampled_tokens = next_tokens.to_numpy()

        # Handle the ragged inputs
        model_inputs = cast(DeepseekV2Inputs, model_inputs)
        tokens = model_inputs.tokens.to_numpy()
        input_row_offsets = model_inputs.input_row_offsets.to_numpy()

        def _get_logits_and_samples(
            batch_index: int, echo: bool
        ) -> tuple[np.ndarray, np.ndarray]:
            if echo:
                start_offset = input_row_offsets[batch_index]
                end_offset = input_row_offsets[batch_index + 1]
                batch_logits = logits[start_offset:end_offset]
                samples = np.concatenate(
                    (
                        tokens[start_offset + 1 : end_offset],
                        sampled_tokens[batch_index : batch_index + 1],
                    )
                )
            else:
                batch_logits = next_token_logits[batch_index : batch_index + 1]
                samples = sampled_tokens[batch_index : batch_index + 1]
            return batch_logits, samples

        return compute_log_probabilities(
            _get_logits_and_samples, batch_top_n, batch_echo
        )
