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
"""Speculative decoding pipelines with factory function and implementations."""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import Device, Tensor, load_devices, scan_available_devices
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    GenerationStatus,
    Pipeline,
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import NullKVCacheManager, PagedKVCacheManager
from max.nn import ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import traced
from transformers import AutoConfig

from ..config_enums import RepoType
from ..hf_utils import download_weight_files
from ..interfaces import (
    GenerateMixin,
    ModelOutputs,
    PipelineModel,
)
from ..sampling import (
    rejection_sampler_with_residuals,
    token_sampler,
)
from ..utils import upper_bounded_default
from .ragged_token_merger import ragged_token_merger

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class SpeculativeDecodingMetrics:
    """Metrics tracker for speculative decoding performance."""

    def __init__(self) -> None:
        """Initialize metrics counters."""
        self.bonus_tokens_used = 0
        self.draft_tokens_accepted = 0
        self.draft_tokens_generated = 0
        self.total_acceptance_lengths = 0
        self.num_generations = 0

    def update(
        self,
        draft_tokens_generated: int,
        draft_tokens_accepted: int,
        bonus_tokens_used: int,
        acceptance_lengths: list[int],
    ) -> None:
        """Update metrics with results from a batch.

        Args:
            draft_tokens_generated: Total draft tokens generated in this batch
            draft_tokens_accepted: Total draft tokens accepted in this batch
            bonus_tokens_used: Number of bonus tokens used in this batch
            acceptance_lengths: List of acceptance lengths for each sequence in batch
        """
        self.draft_tokens_generated += draft_tokens_generated
        self.draft_tokens_accepted += draft_tokens_accepted
        self.bonus_tokens_used += bonus_tokens_used
        self.total_acceptance_lengths += sum(acceptance_lengths)
        self.num_generations += len(acceptance_lengths)

    def get_stats(self) -> dict[str, float]:
        """Get current statistics.

        Returns:
            Dictionary with acceptance rate and total counts
        """
        if self.draft_tokens_generated == 0:
            return {
                "acceptance_rate": 0.0,
                "bonus_tokens_used": 0,
                "draft_tokens_accepted": 0,
                "draft_tokens_generated": 0,
                "avg_acceptance_length": 0.0,
            }

        return {
            "acceptance_rate": self.draft_tokens_accepted
            / self.draft_tokens_generated,
            "bonus_tokens_used": self.bonus_tokens_used,
            "draft_tokens_accepted": self.draft_tokens_accepted,
            "draft_tokens_generated": self.draft_tokens_generated,
            "avg_acceptance_length": self.total_acceptance_lengths
            / self.num_generations
            if self.num_generations > 0
            else 0.0,
        }

    def __str__(self) -> str:
        """String representation of current metrics."""
        stats = self.get_stats()
        return (
            f"SpeculativeDecodingMetrics("
            f"acceptance_rate={stats['acceptance_rate']:.2%}, "
            f"avg_acceptance_length={stats['avg_acceptance_length']:.2f}, "
            f"bonus_tokens_used={stats['bonus_tokens_used']}, "
            f"draft_tokens_accepted={stats['draft_tokens_accepted']}/{stats['draft_tokens_generated']})"
        )


class SpeculativeDecodingPipelineBase(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    GenerateMixin[TextContext, TextGenerationRequest],
    ABC,
):
    """Base class for speculative decoding pipelines with shared logic."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TextContext]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextContext,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
        draft_pipeline_model: type[PipelineModel[TextContext]] | None = None,
        draft_weight_adapters: dict[WeightsFormat, WeightsAdapter]
        | None = None,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._tokenizer = tokenizer

        # Load target model
        self.target_devices = load_devices(
            self.pipeline_config.model_config.device_specs
        )
        target_config = self.pipeline_config.model_config.huggingface_config
        target_session = InferenceSession(devices=self.target_devices)
        self.pipeline_config.configure_session(target_session)
        target_config = AutoConfig.from_pretrained(
            self.pipeline_config.model_config.model_path,
            trust_remote_code=self.pipeline_config.model_config.trust_remote_code,
            revision=self.pipeline_config.model_config.huggingface_model_revision,
        )

        # Expand EOS
        if "eos_token_id" in target_config:
            eos_tokens = target_config.eos_token_id
            if isinstance(eos_tokens, int):
                if eos_tokens != eos_token_id:
                    msg = f"eos_token_id provided in huggingface config ({eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
                    logger.warning(msg)

                self._eos_token_id = set([eos_tokens])
            elif isinstance(eos_tokens, list):
                if eos_token_id in eos_tokens:
                    self._eos_token_id = set(eos_tokens)
                else:
                    self._eos_token_id = set([eos_token_id])
            else:
                msg = f"eos_token_id in huggingface_config, is neither int or list: {eos_tokens}"
                logger.warning(msg)
                self._eos_token_id = set([eos_token_id])
        else:
            self._eos_token_id = set([eos_token_id])

        target_hf_repo = (
            self.pipeline_config.model_config.huggingface_weight_repo
        )

        weight_paths: list[Path] = []
        if (
            self.pipeline_config.model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            weight_paths = download_weight_files(
                huggingface_model_id=target_hf_repo.repo_id,
                filenames=[
                    str(x)
                    for x in self.pipeline_config.model_config.weight_path
                ],
                revision=self.pipeline_config.model_config.huggingface_weight_revision,
                max_workers=8,
            )
        else:
            # Make sure the weight paths are absolute paths
            weight_paths = [
                self.pipeline_config.model_config.model_path / x
                for x in self.pipeline_config.model_config.weight_path
            ]

        target_weights = load_weights(weight_paths)
        _target_weights_format = weights_format(weight_paths)

        if not self.pipeline_config.model_config.quantization_encoding:
            raise ValueError(
                f"quantization_encoding must be provided, {self.pipeline_config.model_config.quantization_encoding}"
            )

        self._target_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=target_session,
            huggingface_config=target_config,
            encoding=self.pipeline_config.model_config.quantization_encoding,
            devices=self.target_devices,
            kv_cache_config=self.pipeline_config.model_config.kv_cache_config,
            weights=target_weights,
            adapter=weight_adapters.get(_target_weights_format),
            return_logits=ReturnLogits.VARIABLE,
        )

        # Calculate Max Length
        self._max_length = self._target_model.calculate_max_seq_len(
            self.pipeline_config,
            huggingface_config=self.pipeline_config.model_config.huggingface_config,
        )

        # Load draft model
        # For now, we are assuming we are placing the draft model will sit
        self.draft_devices = load_devices(scan_available_devices()[:1])
        draft_session = InferenceSession(devices=self.draft_devices)
        self.pipeline_config.configure_session(draft_session)

        assert self.pipeline_config.draft_model_config is not None
        draft_config = (
            self.pipeline_config.draft_model_config.huggingface_config
        )

        if hasattr(
            self.pipeline_config.model_config.huggingface_config, "vocab_size"
        ):
            self.vocab_size = (
                self.pipeline_config.model_config.huggingface_config.vocab_size
            )
        elif hasattr(
            self.pipeline_config.model_config.huggingface_config, "text_config"
        ):
            if hasattr(
                self.pipeline_config.model_config.huggingface_config.text_config,
                "vocab_size",
            ):
                self.vocab_size = self.pipeline_config.model_config.huggingface_config.text_config.vocab_size
            else:
                raise ValueError(
                    "MAXModelConfig's HuggingFace config must have a 'vocab_size' or 'text_config.vocab_size' param for Speculative Decoding"
                )

        else:
            raise ValueError(
                "MAXModelConfig's HuggingFace config must have a 'vocab_size' or 'text_config.vocab_size' param for Speculative Decoding"
            )

        # Retrieve Encoding, and Files for Draft Model
        if self.pipeline_config.draft_model_config is None:
            raise ValueError(
                "draft_model must be provided for speculative decoding"
            )

        draft_hf_repo = (
            self.pipeline_config.draft_model_config.huggingface_weight_repo
        )

        # Use the quantization_encoding from draft_model_config if provided
        if self.pipeline_config.draft_model_config.quantization_encoding:
            draft_encoding = (
                self.pipeline_config.draft_model_config.quantization_encoding
            )
        else:
            # Fall back to first supported encoding if not specified
            encodings = draft_hf_repo.supported_encodings
            if not encodings:
                raise ValueError(
                    "could not identify supported encodings for draft model."
                )
            logger.warning(
                f"using first supported encoding for draft model: {encodings[0]}"
            )
            draft_encoding = encodings[0]

        # Use already-resolved weight paths from draft_model_config
        draft_weight_paths: list[Path] = []
        if (
            self.pipeline_config.draft_model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            draft_weight_paths = download_weight_files(
                huggingface_model_id=self.pipeline_config.draft_model_config.model_path,
                filenames=[
                    str(x)
                    for x in self.pipeline_config.draft_model_config.weight_path
                ],
                revision=self.pipeline_config.draft_model_config.huggingface_weight_revision,
                max_workers=8,
            )
        else:
            # Make sure the weight paths are absolute paths
            draft_weight_paths = [
                self.pipeline_config.draft_model_config.model_path / x
                for x in self.pipeline_config.draft_model_config.weight_path
            ]

        draft_weights = load_weights(draft_weight_paths)
        _draft_weights_format = weights_format(draft_weight_paths)
        assert self.pipeline_config._speculative_config is not None

        # Use draft model's pipeline model and weight adapters if provided
        # Otherwise fall back to target model's (for backward compatibility)
        actual_draft_pipeline_model = (
            draft_pipeline_model
            if draft_pipeline_model is not None
            else pipeline_model
        )
        actual_draft_weight_adapters = (
            draft_weight_adapters
            if draft_weight_adapters is not None
            else weight_adapters
        )

        self._draft_model = actual_draft_pipeline_model(
            pipeline_config=self.pipeline_config,
            session=draft_session,
            huggingface_config=draft_config,
            encoding=draft_encoding,
            devices=self.draft_devices,
            kv_cache_config=self.pipeline_config.draft_model_config.kv_cache_config,
            weights=draft_weights,
            adapter=actual_draft_weight_adapters.get(_draft_weights_format),
            return_logits=ReturnLogits.LAST_TOKEN,
        )

        # Load draft sampler
        draft_sampling_config = self.pipeline_config.sampling_config
        draft_sampling_config.enable_variable_logits = False
        self._draft_sampler = draft_session.load(
            token_sampler(
                draft_sampling_config,
                return_logits=True,
                device=DeviceRef.from_device(self.draft_devices[0]),
            )
        )

        # Load rejection sampler
        self._rejection_sampler = target_session.load(
            rejection_sampler_with_residuals(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

        # Initialize metrics tracker
        self._metrics = SpeculativeDecodingMetrics()

        # Check that the max length for both models are the same
        draft_seq_len = self._draft_model.calculate_max_seq_len(
            self.pipeline_config, draft_config
        )
        target_seq_len = self._target_model.calculate_max_seq_len(
            self.pipeline_config, target_config
        )
        if draft_seq_len != target_seq_len:
            raise ValueError(
                f"draft maximum sequence length ({draft_seq_len}) must match target maximum sequence length."
            )

        self._ragged_token_merger = target_session.load(
            ragged_token_merger(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

        self._draft_session = draft_session
        self._target_session = target_session

        self._num_draft_steps = (
            self.pipeline_config._speculative_config.num_speculative_tokens
        )

    @traced
    def calculate_num_steps(
        self,
        model: PipelineModel[TextContext],
        huggingface_config: AutoConfig,
        num_steps: int,
        context: TextContext,
        is_draft: bool = False,
    ) -> int:
        max_seq_len = model.calculate_max_seq_len(
            self.pipeline_config, huggingface_config=huggingface_config
        )
        if is_draft:
            max_seq_len -= 1
        num_available_steps = context.compute_num_available_steps(max_seq_len)

        if num_available_steps <= 0:
            raise ValueError(
                f"Request {context.request_id} length ({context.current_length}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return min(num_available_steps, num_steps)

    @property
    def pipeline_config(self) -> PipelineConfig:
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        return self._tokenizer

    def _create_sampling_parameters(
        self,
        batch: list[TextContext],
        device: Device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Create sampling parameter tensors from context batch.

        Args:
            batch: List of context objects containing sampling parameters
            device: Device to place the tensors on

        Returns:
            Tuple of (top_k, max_k, temperature, top_p, min_top_p, seed) tensors
        """
        top_k_np = np.array(
            [context.sampling_params.top_k for context in batch], dtype=np.int64
        )
        top_k = Tensor.from_numpy(top_k_np).to(device)
        max_k_np = np.array(np.max(top_k_np), dtype=np.int64)
        max_k = Tensor.from_numpy(max_k_np)
        temperature_np = np.array(
            [context.sampling_params.temperature for context in batch],
            dtype=np.float32,
        )
        temperature = Tensor.from_numpy(temperature_np).to(device)
        top_p_np = np.array(
            [context.sampling_params.top_p for context in batch],
            dtype=np.float32,
        )
        top_p = Tensor.from_numpy(top_p_np).to(device)
        # min_top_p must be provided as a scalar CPU tensor
        min_top_p_np = np.array(np.min(top_p_np), dtype=np.float32)
        min_top_p = Tensor.from_numpy(min_top_p_np)
        seed_np = np.array(
            [context.sampling_params.seed for context in batch], dtype=np.uint64
        )
        seed = Tensor.from_numpy(seed_np).to(device)

        return (top_k, max_k, temperature, top_p, min_top_p, seed)

    @traced
    def sample_draft_logits(
        self,
        model_outputs: ModelOutputs,
        prev_tokens: Tensor,
        prev_logits: Tensor,
        top_k: Tensor,
        max_k: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        min_top_p: Tensor,
        seed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        graph_inputs = [
            model_outputs.logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
            prev_logits,
        ]
        a, b, c = self._draft_sampler(*graph_inputs)[:3]
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        assert isinstance(c, Tensor)
        return (a, b, c)

    @property
    def kv_managers(
        self,
    ) -> list[PagedKVCacheManager | NullKVCacheManager]:
        return [self._draft_model.kv_manager, self._target_model.kv_manager]

    @property
    def metrics(self) -> SpeculativeDecodingMetrics:
        """Get the current speculative decoding metrics.

        Returns:
            The SpeculativeDecodingMetrics instance with current statistics
        """
        return self._metrics

    def __del__(self) -> None:
        """Log metrics when the pipeline is destroyed."""
        if (
            hasattr(self, "_metrics")
            and self._metrics.draft_tokens_generated > 0
        ):
            logger.info(f"Speculative decoding metrics: {self._metrics}")

    def update_contexts(
        self,
        context_batch: list[TextContext],
        first_rejected_tokens: npt.NDArray[np.integer[Any]],
        recovered_tokens: npt.NDArray[np.integer[Any]],
        bonus_tokens: npt.NDArray[np.integer[Any]],
        draft_tokens: npt.NDArray[np.integer[Any]],
        num_draft_tokens_generated: int,
    ) -> None:
        """Update contexts with the results of token generation.

        Args:
            context_batch: The list of context objects
            first_rejected_tokens: Array indicating the indices of first rejected tokens
            recovered_tokens: Array of recovered tokens from target model
            bonus_tokens: Array of bonus tokens from target model
            draft_tokens: Array of draft tokens
            num_draft_tokens_generated: Number of tokens generated by the draft model
        """
        total_draft_generated = num_draft_tokens_generated * len(context_batch)
        total_draft_accepted = 0
        total_bonus_used = 0
        acceptance_lengths = []

        for idx, rejected_token_idx in enumerate(first_rejected_tokens):
            context = context_batch[idx]
            rejected_token_idx = rejected_token_idx.item()

            context.bump_token_indices(
                active_idx=-num_draft_tokens_generated,
                end_idx=-num_draft_tokens_generated,
            )

            for token_idx in range(rejected_token_idx):
                token = int(draft_tokens[idx, token_idx])
                context.update(token)

            if rejected_token_idx == num_draft_tokens_generated:
                context.update(bonus_tokens[idx, 0].item())
                total_bonus_used += 1
            else:
                context.update(recovered_tokens[idx, rejected_token_idx].item())

            total_draft_accepted += rejected_token_idx
            acceptance_lengths.append(rejected_token_idx)

            # When some or all draft tokens are rejected, we apply a token from
            # the residual distribution. The draft and target models have not
            # processed this token so the context goes back one step for both
            # of the models to process that token.
            # If all draft tokens are accepted, then the draft model has not
            # processed the bonus token. In this case only the draft needs to
            # go one step back. At the moment we do this for all cases.
            context.bump_token_indices(start_idx=-1)

        # Update metrics
        self._metrics.update(
            total_draft_generated,
            total_draft_accepted,
            total_bonus_used,
            acceptance_lengths,
        )

    def build_response(
        self, context_batch: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        """Build response from updated contexts.

        Args:
            context_batch: The list of context objects

        Returns:
            Dictionary mapping request IDs to TextGenerationOutput objects
        """
        res: dict[RequestID, TextGenerationOutput] = {}

        for context in context_batch:
            # Identify the Max Length
            context_max_length = upper_bounded_default(
                upper_bound=self._max_length, default=context.max_length
            )

            # Break early if beyond max length
            current_length = context.start_idx + 1
            if current_length >= context_max_length:
                context.status = GenerationStatus.MAXIMUM_LENGTH

            # Construct generation output
            res[context.request_id] = context.to_generation_output()

        return res

    @traced
    def release(self, request_id: RequestID) -> None:
        """Releases resources associated with this request ID.

        Args:
            request_id: Unique identifier for the finished request.

        """
        self._draft_model.kv_manager.release(request_id)
        self._target_model.kv_manager.release(request_id)
