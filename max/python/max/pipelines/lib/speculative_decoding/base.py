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
"""Speculative decoding pipelines with factory function and implementations."""

from __future__ import annotations

import logging
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DLPackArray, load_devices
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
    PipelineTokenizer,
    RequestID,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import traced
from transformers import AutoConfig

from ..config_enums import RepoType
from ..hf_utils import download_weight_files
from ..interfaces import ModelOutputs, PipelineModel
from ..pipeline_variants.text_generation import TextGenerationPipelineInterface
from ..sampling import rejection_sampler_with_residuals, token_sampler
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


def hidden_states_return_config(
    pipeline_config: PipelineConfig, is_draft: bool = False
) -> ReturnHiddenStates:
    """Return the hidden states return config for the speculative config.

    For Eagle and DeepSeek MTP, we share the embedding and lm_head weights between the target and draft models and only take the last hidden state from the target model.

    """
    assert pipeline_config.speculative is not None
    is_eagle_or_mtp = (
        pipeline_config.speculative.is_eagle()
        or pipeline_config.speculative.is_mtp()
    )
    if is_eagle_or_mtp:
        if is_draft:
            return ReturnHiddenStates.LAST
        else:
            return ReturnHiddenStates.ALL_NORMALIZED

    else:
        return ReturnHiddenStates.NONE


class SpeculativeDecodingPipelineBase(
    TextGenerationPipelineInterface[TextContext],
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
            self.pipeline_config.model.device_specs
        )
        target_config = self.pipeline_config.model.huggingface_config
        target_session = InferenceSession(devices=self.target_devices)
        self.pipeline_config.configure_session(target_session)
        target_config = AutoConfig.from_pretrained(
            self.pipeline_config.model.model_path,
            trust_remote_code=self.pipeline_config.model.trust_remote_code,
            revision=self.pipeline_config.model.huggingface_model_revision,
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

        target_hf_repo = self.pipeline_config.model.huggingface_weight_repo

        weight_paths: list[Path] = []
        if (
            self.pipeline_config.model.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            weight_paths = download_weight_files(
                huggingface_model_id=target_hf_repo.repo_id,
                filenames=[
                    str(x) for x in self.pipeline_config.model.weight_path
                ],
                revision=self.pipeline_config.model.huggingface_weight_revision,
                max_workers=8,
            )
        else:
            # Use the resolved repo_id (which points to local cache in offline mode)
            local_path = Path(target_hf_repo.repo_id)
            weight_paths = [
                local_path / x for x in self.pipeline_config.model.weight_path
            ]

        target_weights = load_weights(weight_paths)
        _target_weights_format = weights_format(weight_paths)

        if not self.pipeline_config.model.quantization_encoding:
            raise ValueError(
                f"quantization_encoding must be provided, {self.pipeline_config.model.quantization_encoding}"
            )

        self._target_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=target_session,
            huggingface_config=target_config,
            encoding=self.pipeline_config.model.quantization_encoding,
            devices=self.target_devices,
            kv_cache_config=self.pipeline_config.model.kv_cache,
            weights=target_weights,
            adapter=weight_adapters.get(_target_weights_format),
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=False
            ),
        )

        # Validate that target model has HuggingFace config
        target_hf_config = self.pipeline_config.model.huggingface_config
        if target_hf_config is None:
            raise ValueError(
                f"Speculative decoding requires a HuggingFace config for the target model, "
                f"but could not load config for '{self.pipeline_config.model.model_path}'. "
                "Please ensure the target model is a standard Transformers model with a valid config.json."
            )

        # Calculate Max Length
        self._max_length = self._target_model.calculate_max_seq_len(
            self.pipeline_config,
            huggingface_config=target_hf_config,
        )

        # Load draft model
        assert self.pipeline_config.draft_model is not None
        self.draft_devices = load_devices(
            self.pipeline_config.draft_model.device_specs
        )
        draft_session = InferenceSession(devices=self.draft_devices)
        self.pipeline_config.configure_session(draft_session)

        if self.pipeline_config.draft_model is None:
            raise ValueError("Draft model is required for speculative decoding")
        draft_config = self.pipeline_config.draft_model.huggingface_config
        if draft_config is None:
            raise ValueError(
                f"Speculative decoding requires a HuggingFace config for the draft model, "
                f"but could not load config for '{self.pipeline_config.draft_model.model_path}'. "
                "Please ensure the draft model is a standard Transformers model with a valid config.json."
            )

        if hasattr(target_hf_config, "vocab_size"):
            self.vocab_size = target_hf_config.vocab_size
        elif hasattr(target_hf_config, "text_config"):
            if hasattr(target_hf_config.text_config, "vocab_size"):
                self.vocab_size = target_hf_config.text_config.vocab_size
            else:
                raise ValueError(
                    "MAXModelConfig's HuggingFace config must have a 'vocab_size' or 'text_config.vocab_size' param for Speculative Decoding"
                )

        else:
            raise ValueError(
                "MAXModelConfig's HuggingFace config must have a 'vocab_size' or 'text_config.vocab_size' param for Speculative Decoding"
            )

        # Retrieve Encoding, and Files for Draft Model
        if self.pipeline_config.draft_model is None:
            raise ValueError(
                "draft_model must be provided for speculative decoding"
            )

        draft_hf_repo = self.pipeline_config.draft_model.huggingface_weight_repo

        # Use the quantization_encoding from draft_model if provided
        if self.pipeline_config.draft_model.quantization_encoding:
            draft_encoding = (
                self.pipeline_config.draft_model.quantization_encoding
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

        # Use already-resolved weight paths from draft_model
        draft_weight_paths: list[Path] = []
        if (
            self.pipeline_config.draft_model.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            draft_weight_paths = download_weight_files(
                huggingface_model_id=self.pipeline_config.draft_model.model_path,
                filenames=[
                    str(x) for x in self.pipeline_config.draft_model.weight_path
                ],
                revision=self.pipeline_config.draft_model.huggingface_weight_revision,
                max_workers=8,
            )
        else:
            # Use the resolved repo_id (which points to local cache in offline mode)
            draft_local_path = Path(
                self.pipeline_config.draft_model.huggingface_weight_repo.repo_id
            )
            draft_weight_paths = [
                draft_local_path / x
                for x in self.pipeline_config.draft_model.weight_path
            ]

        draft_weights = load_weights(draft_weight_paths)
        _draft_weights_format = weights_format(draft_weight_paths)
        assert self.pipeline_config.speculative is not None
        self._speculative_config = self.pipeline_config.speculative

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

        shared_weights: dict[str, DLPackArray] | None = None
        shared_ep_comm_initializer = None
        if self.pipeline_config.speculative is not None and (
            self.pipeline_config.speculative.is_eagle()
            or self.pipeline_config.speculative.is_mtp()
        ):
            shared_weights, shared_ep_comm_initializer = (
                self._maybe_build_deepseekv3_nextn_shared_resources(
                    actual_draft_pipeline_model
                )
            )

        draft_model_kwargs: dict[str, Any] = {}
        if shared_weights and getattr(
            actual_draft_pipeline_model, "supports_shared_weights", False
        ):
            draft_model_kwargs["shared_weights"] = shared_weights
        elif shared_weights:
            logger.debug(
                "Draft model %s does not support shared weights; skipping",
                actual_draft_pipeline_model.__name__,
            )

        if shared_ep_comm_initializer is not None:
            draft_model_kwargs["shared_ep_comm_initializer"] = (
                shared_ep_comm_initializer
            )

        self._draft_model = actual_draft_pipeline_model(
            pipeline_config=self.pipeline_config,
            session=draft_session,
            huggingface_config=draft_config,
            encoding=draft_encoding,
            devices=self.draft_devices,
            kv_cache_config=self.pipeline_config.draft_model.kv_cache,
            weights=draft_weights,
            adapter=actual_draft_weight_adapters.get(_draft_weights_format),
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=True
            ),
            **draft_model_kwargs,
        )

        # Load draft sampler
        draft_sampling_config = self.pipeline_config.sampling
        draft_sampling_config.enable_variable_logits = False
        self._draft_sampler = draft_session.load(
            token_sampler(
                draft_sampling_config,
                return_logits=True,
                device=DeviceRef.from_device(self.draft_devices[0]),
            )
        )

        # TODO: add option to load greedy sampler
        self._rejection_sampler = target_session.load(
            rejection_sampler_with_residuals(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

        # Initialize metrics tracker
        self._metrics = SpeculativeDecodingMetrics()

        # Track draft model replica assignments per request
        self._draft_replica_idx: dict[RequestID, int] = {}

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
            self.pipeline_config.speculative.num_speculative_tokens
        )

    def _maybe_build_deepseekv3_nextn_shared_resources(
        self,
        draft_model_cls: type[PipelineModel[TextContext]],
    ) -> tuple[dict[str, DLPackArray] | None, Any]:
        # Imported here to avoid circular imports
        from max.pipelines.architectures.deepseekV3.model import (  # type: ignore[import-not-found]
            DeepseekV3Model,
        )
        from max.pipelines.architectures.deepseekV3_nextn.model import (  # type: ignore[import-not-found]
            DeepseekV3NextNModel,
        )

        if not isinstance(self._target_model, DeepseekV3Model):
            return None, None
        if not issubclass(draft_model_cls, DeepseekV3NextNModel):
            return None, None

        # Share EP buffers between target and draft to avoid duplicating
        shared_ep_comm_initializer = self._target_model.ep_comm_initializer
        target_state_dict = getattr(self._target_model, "state_dict", None)
        if not isinstance(target_state_dict, dict):
            raise ValueError(
                "Target DeepseekV3 model has no state_dict; "
                "cannot share weights with NextN draft model."
            )

        required_prefixes = ("embed_tokens.", "lm_head.")
        shared_weights: dict[str, DLPackArray] = {}
        for name, value in target_state_dict.items():
            for prefix in required_prefixes:
                if name.startswith(prefix):
                    shared_weights[name] = value

        if len(shared_weights) != len(required_prefixes):
            raise ValueError(
                f"Missing weight prefixes {required_prefixes} in target DeepseekV3 "
                f"state_dict. Cannot share weights with NextN draft model."
            )

        logger.info(
            "Sharing DeepseekV3 embedding and head weights with NextN draft model."
        )
        return shared_weights, shared_ep_comm_initializer

    @traced
    def calculate_num_steps(
        self,
        model: PipelineModel[TextContext],
        huggingface_config: AutoConfig,
        num_steps: int,
        context: TextContext,
        is_draft: bool = False,
    ) -> int:
        """Computes the number of steps to run for the given context."""
        max_seq_len = model.calculate_max_seq_len(
            self.pipeline_config, huggingface_config=huggingface_config
        )
        if is_draft:
            max_seq_len -= 1
        num_available_steps = context.compute_num_available_steps(max_seq_len)

        if num_available_steps <= 0:
            raise ValueError(
                f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return min(num_available_steps, num_steps)

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Returns the tokenizer for this speculative pipeline."""
        return self._tokenizer

    def _create_sampling_parameters(
        self,
        batch: list[TextContext],
        device: Device,
    ) -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer]:
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
        top_k = Buffer.from_numpy(top_k_np).to(device)
        max_k_np = np.array(np.max(top_k_np), dtype=np.int64)
        max_k = Buffer.from_numpy(max_k_np)
        temperature_np = np.array(
            [context.sampling_params.temperature for context in batch],
            dtype=np.float32,
        )
        temperature = Buffer.from_numpy(temperature_np).to(device)
        top_p_np = np.array(
            [context.sampling_params.top_p for context in batch],
            dtype=np.float32,
        )
        top_p = Buffer.from_numpy(top_p_np).to(device)
        # min_top_p must be provided as a scalar CPU tensor
        min_top_p_np = np.array(np.min(top_p_np), dtype=np.float32)
        min_top_p = Buffer.from_numpy(min_top_p_np)
        seed_np = np.array(
            [context.sampling_params.seed for context in batch], dtype=np.uint64
        )
        seed = Buffer.from_numpy(seed_np).to(device)

        return (top_k, max_k, temperature, top_p, min_top_p, seed)

    @traced
    def sample_draft_logits(
        self,
        model_outputs: ModelOutputs,
        prev_tokens: Buffer,
        prev_logits: Buffer,
        top_k: Buffer,
        max_k: Buffer,
        temperature: Buffer,
        top_p: Buffer,
        min_top_p: Buffer,
        seed: Buffer,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Samples draft tokens from the draft model logits."""
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
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return (a, b, c)

    @property
    def kv_managers(
        self,
    ) -> list[PagedKVCacheManager]:
        """Returns the KV cache managers for target and draft models."""
        return [self._target_model.kv_manager, self._draft_model.kv_manager]

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
            context.tokens.rewind_processing(1)

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
            current_length = context.tokens.processed_length + 1
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

        Note: Target model KV cache is released by the scheduler via batch_constructor.
        This method only releases the draft model KV cache, which the scheduler
        doesn't know about.
        """
        # Release draft model KV cache (scheduler doesn't manage this).
        # The request may not have been claimed yet if it errored before
        # execute() ran the draft model, so check before releasing.
        replica_idx = self._draft_replica_idx.pop(request_id, 0)
        if self._draft_model.kv_manager.contains(
            request_id, replica_idx=replica_idx
        ):
            self._draft_model.kv_manager.release(
                request_id, replica_idx=replica_idx
            )
        # Target model KV cache is released by scheduler via batch_constructor
