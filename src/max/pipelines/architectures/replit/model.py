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
import time
import warnings
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
from max.driver import Device, DeviceSpec, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import GGUFWeights, Weights, WeightsAdapter
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
    upper_bounded_default,
)
from max.pipelines.interfaces import LogProbabilities
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.log_probabilities import compute_log_probabilities
from transformers import AutoConfig

from .graph import _build_graph
from .model_config import ReplitConfig

logger = logging.getLogger("max.pipelines")


class ReplitInputs(ModelInputs):
    """A class representing inputs for the Replit model.

    This class encapsulates the input tensors required for the Replit model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    """

    tokens: Tensor
    input_row_offsets: Tensor

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.kv_cache_inputs = kv_cache_inputs


class ReplitModel(PipelineModel[TextContext]):
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
    ) -> None:
        if pipeline_config.model_config.device_specs[0] == DeviceSpec.cpu():
            msg = "Replit currently only supported on gpu."
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
        )
        self.model = self.load_model(session)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(ReplitInputs, model_inputs)
        # keep mypy happy.
        assert model_inputs.kv_cache_inputs is not None, (
            "Replit has KV cache inputs"
        )
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            *model_inputs.kv_cache_inputs,
            copy_inputs_to_device=False,
        )
        if self.pipeline_config.enable_echo:
            assert len(model_outputs) == 2
            assert isinstance(model_outputs[0], Tensor)
            assert isinstance(model_outputs[1], Tensor)
            return ModelOutputs(
                next_token_logits=model_outputs[0], logits=model_outputs[1]
            )
        else:
            assert len(model_outputs) == 1
            assert isinstance(model_outputs[0], Tensor)
            return ModelOutputs(next_token_logits=model_outputs[0])

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> ReplitInputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        if kv_cache_inputs is None:
            raise ValueError(
                "Replit has KV cache inputs, but got None instead."
            )
        return ReplitInputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> ReplitInputs:
        prev_model_inputs = cast(ReplitInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]
        return ReplitInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return ReplitConfig.get_num_layers(huggingface_config)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return ReplitConfig.get_kv_params(
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
                upper_bound=huggingface_config.max_seq_len,
                default=pipeline_config.max_length,
            )
        except ValueError as e:
            msg = (
                "Unable to infer max_length for Replit, the provided "
                f"max_length ({pipeline_config.max_length}) exceeds the "
                f"model's max_seq_len "
                f"({huggingface_config.max_seq_len})."
            )
            raise ValueError(msg) from e

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=ReplitConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.huggingface_config.n_layers,
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
            params=ReplitConfig.get_kv_params(
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
            num_layers=huggingface_config.n_layers,
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
        if not isinstance(self.weights, GGUFWeights):
            msg = "only gguf weights supported in Replit."
            raise ValueError(msg)

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = _build_graph(
            self.pipeline_config,
            self.weights,
            ReplitConfig.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            kv_manager=self.kv_manager,
            huggingface_config=self.huggingface_config,
            dtype=self.dtype,
        )
        model = session.load(
            graph, weights_registry=self.weights.allocated_weights
        )
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
        model_inputs = cast(ReplitInputs, model_inputs)
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
