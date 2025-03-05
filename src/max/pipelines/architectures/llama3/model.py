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

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, List, Literal, Sequence, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import WeightData, Weights
from max.pipelines import (
    LogProbabilities,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    RopeType,
    SupportedEncoding,
    TextContext,
    WeightsFormat,
    upper_bounded_default,
)
from max.pipelines.dataprocessing import batch_padded_tokens_and_mask
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.nn import LayerV2, Llama3RopeScalingParams, Signals
from max.pipelines.nn.compute_log_probabilities import compute_log_probabilities
from transformers import AutoConfig

from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config
from .naive_llama3 import NaiveLlama3

logger = logging.getLogger("max.pipelines")


class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: np.ndarray | Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets_or_attn_mask: np.ndarray | Tensor
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        tokens: np.ndarray | Tensor,
        input_row_offsets_or_attn_mask: np.ndarray | Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets_or_attn_mask: Input row offsets (ragged tensors)
                or attention mask (padded tensors).
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
        """
        self.tokens = tokens
        self.input_row_offsets_or_attn_mask = input_row_offsets_or_attn_mask
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs

    @property
    def input_row_offsets(self) -> np.ndarray | Tensor:
        """Gets the row offsets of the ragged input sequence."""
        # TODO(bduke): this should implement a ragged tensor interface.
        return self.input_row_offsets_or_attn_mask


class LlamaModelBase(PipelineModel[TextContext]):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    """Postprocessor for the logits."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
        super().__init__(pipeline_config, session, huggingface_config)
        self.model = self.load_model(session)

        # Initialize state needed for communication collectives.
        self.signal_buffers = (
            [
                Tensor.zeros(
                    shape=(Signals.NUM_BYTES,),
                    dtype=DType.uint8,
                    device=dev,
                )
                for dev in pipeline_config.devices
            ]
            if len(pipeline_config.devices) > 1
            # Skip creating buffers for single-device, where communication
            # collectives shouldn't be called.
            else []
        )

    @classmethod
    def get_kv_params(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=pipeline_config.cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=(
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            ),
            page_size=pipeline_config.kv_cache_config.kv_cache_page_size,
            cache_strategy=pipeline_config.kv_cache_config.cache_strategy,
            enable_prefix_caching=pipeline_config.kv_cache_config.enable_prefix_caching,
            n_devices=len(pipeline_config.devices),
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(Llama3Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets_or_attn_mask,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            copy_inputs_to_device=(
                not self.pipeline_config.kv_cache_config.cache_strategy.uses_opaque()
            ),
        )

        if self.pipeline_config.enable_echo:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[1]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0])
            )

    def _prepare_ragged_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return Llama3Inputs(
            tokens=Tensor.from_numpy(tokens).to(
                self.pipeline_config.devices[0]
            ),
            input_row_offsets_or_attn_mask=Tensor.from_numpy(
                input_row_offsets
            ).to(self.pipeline_config.devices[0]),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def _prepare_padded_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        return Llama3Inputs(
            tokens=next_tokens_batch,
            input_row_offsets_or_attn_mask=attn_mask,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        if self.pipeline_config.kv_cache_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_initial_token_inputs(
                context_batch, kv_cache_inputs
            )
        else:
            return self._prepare_padded_initial_token_inputs(
                context_batch, kv_cache_inputs
            )

    def _prepare_ragged_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: Llama3Inputs,
    ) -> Llama3Inputs:
        row_offsets_size = (
            prev_model_inputs.input_row_offsets_or_attn_mask.shape[0]
        )
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return Llama3Inputs(
            tokens=next_tokens,
            input_row_offsets_or_attn_mask=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Llama3Inputs:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        prev_model_inputs = cast(Llama3Inputs, prev_model_inputs)
        if self.pipeline_config.kv_cache_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_next_token_inputs(
                next_tokens, prev_model_inputs
            )
        else:
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            msg = "multistep unsupported for padded token batches"
            raise ValueError(msg)

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
                "Unable to infer max_length for Llama3, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self.get_kv_params(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.pipeline_config.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.pipeline_config.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: List[Device],
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=cls.get_kv_params(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=cls.get_num_layers(
                huggingface_config=huggingface_config,
            ),
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
        ).to(self.pipeline_config.devices[0])

        # Read in weights.
        self._weights = self.pipeline_config.load_weights()

        if serialized_path := self.pipeline_config.serialized_model_path:
            # Hydrate all weights to be referenced by the serialized path.
            weights_registry = {}
            for name, weight in self._weights.items():
                weights_registry[name] = weight.raw_tensor()

            logger.info("Loading serialized model from %s", serialized_path)

            return session.load(
                serialized_path, weights_registry=weights_registry
            )

        else:
            logger.info("Building and compiling model...")
            before = time.perf_counter()
            graph = self._build_graph(self._weights)
            model = session.load(graph, weights_registry=self.state_dict)
            after = time.perf_counter()
            logger.info(
                f"Building and compiling model took {after - before:.6f} seconds"
            )
            if (
                export_path
                := self.pipeline_config.save_to_serialized_model_path
            ):
                logger.info("Exporting serialized model to %s", export_path)
                model._export_mef(export_path)
            return model

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> List[tuple[TensorValue, ...]]:
        kv_params = self.get_kv_params(
            self.pipeline_config, huggingface_config=self.huggingface_config
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev = [
            tuple(
                kv_inputs_flat[
                    i * len_of_kv_tuple_per_dev : (i + 1)
                    * len_of_kv_tuple_per_dev
                ]
            )
            for i in range(n_devices)
        ]
        return kv_caches_per_dev

    @property
    def _attention_multiplier(self) -> float:
        """The attention multiplier is a scalar that scales the attention scores.
        It is used to control the variance of the attention scores.

        This function is used to get the attention multiplier from the
        huggingface config. If the attention multiplier is not set, it will be
        calculated as the square root of 1.0 divided by the head dimension.
        """
        return getattr(
            self.huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0
                / self.get_kv_params(
                    self.pipeline_config,
                    huggingface_config=self.huggingface_config,
                ).head_dim
            ),
        )

    def _build_opaque_graph(self, weights: Weights) -> Graph:
        device0 = self.pipeline_config.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        huggingface_config = self.huggingface_config
        adapter = self.pipeline_config._weight_adapters.get(
            self.pipeline_config.weights_format
        )
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        model_config = self._model_config(state_dict)
        nn_model: LayerV2
        if len(self.pipeline_config.devices) > 1:
            kv_cache_args = self.kv_manager.input_symbols()
            flattened_kv_types = [
                kv_type for sublist in kv_cache_args for kv_type in sublist
            ]

            # Create metadata for signal buffers.
            signals = Signals(
                devices=(
                    DeviceRef(d.label, d.id)
                    for d in self.pipeline_config.devices
                )
            )

            nn_model = DistributedLlama3(model_config)
            nn_model.load_state_dict(state_dict)
            self.state_dict = nn_model.state_dict()
            with Graph(
                getattr(self.huggingface_config, "model_type", "llama3"),
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *signals.input_types(),
                    *flattened_kv_types,
                ],
            ) as graph:
                tokens, input_row_offsets, *variadic_args = graph.inputs

                # Multi-GPU passes a signal buffer per device: unmarshal those.
                signal_buffers = [
                    v.buffer
                    for v in variadic_args[: len(self.pipeline_config.devices)]
                ]

                # Unmarshal the remaining arguments, which are for KV cache.
                kv_cache = [
                    v.tensor
                    for v in variadic_args[len(self.pipeline_config.devices) :]
                ]

                kv_caches_per_dev = self._unflatten_kv_inputs(kv_cache)

                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph
        else:
            nn_model = Llama3(model_config)
            nn_model.load_state_dict(state_dict)
            self.state_dict = nn_model.state_dict()
            with Graph(
                "llama3",
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *self.kv_manager.input_symbols()[0],
                ],
            ) as graph:
                tokens, input_row_offsets, *kv_cache_inputs = graph.inputs
                outputs = nn_model(
                    tokens.tensor,
                    [inp.tensor for inp in kv_cache_inputs],
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph

    def _build_graph(self, weights: Weights) -> Graph:
        if self.pipeline_config.kv_cache_config.cache_strategy.uses_opaque():
            return self._build_opaque_graph(weights)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        if len(self.pipeline_config.devices) > 1:
            raise ValueError(
                "Naive mode does not support distributed execution"
            )

        kv_inputs = self.kv_manager.input_symbols()[0]

        adapter = self.pipeline_config._weight_adapters.get(
            self.pipeline_config.weights_format
        )
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        model_config = self._model_config(state_dict)
        nn_model = NaiveLlama3(model_config)

        # Load weights. We allow the weight types to be overriden due to
        # multiple quantization enodings in GGUF checkpoints.
        nn_model.load_state_dict(
            state_dict, override_quantization_encoding=True
        )
        self.state_dict = nn_model.state_dict()

        with Graph(
            getattr(self.huggingface_config, "model_type", "llama3"),
            input_types=[
                tokens_type,
                attn_mask_type,
                *kv_inputs,
            ],
        ) as graph:
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
            mask_dtype = (
                self.pipeline_config.dtype
                if self.pipeline_config.quantization_encoding
                in [
                    SupportedEncoding.float32,
                    SupportedEncoding.bfloat16,
                ]
                else (
                    DType.float32
                    if self.pipeline_config.devices[0].label == "cpu"
                    else DType.bfloat16
                )
            )
            logits = nn_model(
                tokens.tensor,
                attention_mask.tensor.cast(mask_dtype),
                k_cache.buffer,
                v_cache.buffer,
                start_pos.tensor,
            )[0]

            if self.pipeline_config.enable_echo:
                graph.output(logits[:, -1], logits)
            else:
                graph.output(logits[:, -1])

            return graph

    def _model_config(self, state_dict: dict[str, WeightData]):
        huggingface_config = self.pipeline_config.huggingface_config
        interleaved_rope_weights = (
            self.pipeline_config.weights_format == WeightsFormat.gguf
            and self.pipeline_config.rope_type == RopeType.normal
        )
        rms_norm_eps = None
        if self.norm_method == "rms_norm":
            if huggingface_config.model_type == "exaone":
                rms_norm_eps = huggingface_config.layer_norm_epsilon
            else:
                rms_norm_eps = huggingface_config.rms_norm_eps

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in self.pipeline_config.device_specs
        ]

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "lm_head.weight" not in state_dict
        )
        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )
        rope_scaling_params = None
        rope_scaling = huggingface_config.rope_scaling
        if rope_scaling is not None and rope_scaling["rope_type"] == "llama3":
            rope_scaling_params = Llama3RopeScalingParams(
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                orig_max_position=rope_scaling[
                    "original_max_position_embeddings"
                ],
            )

        return Llama3Config(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=huggingface_config.rope_theta,
            rope_scaling_params=rope_scaling_params,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=self.pipeline_config.dtype,
            quantization_encoding=self.pipeline_config.graph_quantization_encoding,
            quantization_config=self.pipeline_config._quant_config,
            all_logits=self.pipeline_config.enable_echo,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            kv_params=self.get_kv_params(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            norm_method=self.norm_method,
            tie_word_embeddings=tie_word_embeddings,
            stacked_mlp="layers.0.mlp.gate_up_proj.weight" in state_dict,
            stacked_qkv="layers.0.self_attn.qkv_proj.weight" in state_dict,
            logits_postprocessor=self.logits_postprocessor,
            attention_multiplier=self._attention_multiplier,
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            clip_qkv=getattr(
                self.pipeline_config.huggingface_config, "clip_qkv", None
            ),
        )

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
                logger.warning(
                    "Could not get logprobs with echo because the full logits"
                    f" were not returned by {self.pipeline_config.model_path}"
                    " model. Please ensure that this model is started with "
                    "`--enable-echo`."
                )
                assert not self.pipeline_config.enable_echo, (
                    "Echo was enabled but logits were not returned."
                )
                return None
            logits = model_outputs.logits.to_numpy()

        llama3_inputs = cast(Llama3Inputs, model_inputs)
        next_token_logits = cast(
            Tensor, model_outputs.next_token_logits
        ).to_numpy()

        sampled_tokens = next_tokens.to_numpy()
        if self.pipeline_config.kv_cache_config.cache_strategy.uses_opaque():
            # Handle the ragged inputs
            tokens = cast(Tensor, llama3_inputs.tokens).to_numpy()
            input_row_offsets = cast(
                Tensor, llama3_inputs.input_row_offsets
            ).to_numpy()

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
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        else:
            # Handle batched inputs. Llama pads them to the right so the seq
            # lengths can be computed by finding the first 0 token.
            tokens = cast(np.ndarray, llama3_inputs.tokens)
            seq_lens = np.sum(tokens > 0, axis=1)

            def _get_logits_and_samples(
                batch_index: int, echo: bool
            ) -> tuple[np.ndarray, np.ndarray]:
                if echo:
                    seq_len = seq_lens[batch_index]
                    padded_tokens = tokens[batch_index]

                    batch_logits = logits[batch_index, :seq_len, :]
                    samples = np.concatenate(
                        (
                            padded_tokens[1:seq_len],
                            sampled_tokens[batch_index : batch_index + 1],
                        )
                    )
                else:
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1, :
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        return compute_log_probabilities(
            _get_logits_and_samples, batch_top_n, batch_echo
        )


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
    ) -> None:
        super().__init__(pipeline_config, session, huggingface_config)
