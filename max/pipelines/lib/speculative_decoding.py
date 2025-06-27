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
# mypy: disable-error-code="import-not-found"
"""Speculative Decoding Text Generation Pipeline"""

import logging
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

import numpy as np
from max.driver import Tensor, load_devices, scan_available_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputsSequence
from max.pipelines.core import (
    InputContext,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
)
from max.profiler import traced
from transformers import AutoConfig

from .config_enums import RepoType
from .hf_utils import download_weight_files
from .pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    upper_bounded_default,
)
from .ragged_token_merger import ragged_token_merger
from .sampling import rejection_sampler_with_residuals, token_sampler

T = TypeVar("T", bound=InputContext)

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


class SpeculativeDecodingTextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline with speculative decoding."""

    def __init__(
        self,
        pipeline_config: Any,  # PipelineConfig
        pipeline_model: type[PipelineModel],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
    ) -> None:
        self.pipeline_config = pipeline_config

        # Load target model
        self.target_devices = load_devices(
            self.pipeline_config.model_config.device_specs
        )
        target_config = self.pipeline_config.model_config.huggingface_config
        target_session = InferenceSession(devices=self.target_devices)
        target_session.gpu_profiling(
            self.pipeline_config.profiling_config.gpu_profiling
        )
        target_config = AutoConfig.from_pretrained(
            self.pipeline_config.model_config.model_path,
            trust_remote_code=self.pipeline_config.model_config.trust_remote_code,
            revision=self.pipeline_config.model_config.huggingface_model_revision,
        )

        # Expand EOS
        if pipeline_config.ignore_eos:
            self._eos_token_id = set([])
        elif "eos_token_id" in target_config:
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
            adapter=weight_adapters.get(_target_weights_format, None),
            return_logits=ReturnLogits.VARIABLE,
        )
        # TODO: We only support Llama3 for spec decoding since we haven't worked out
        # a general API for the prepare next token with draft inputs yet.
        from ..architectures.llama3.model import Llama3Model

        # Now check if the instantiated model is a Llama3Model
        if not isinstance(self._target_model, Llama3Model):
            raise ValueError(
                "Speculative decoding only supported for Llama3 models"
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
        draft_session.gpu_profiling(
            self.pipeline_config.profiling_config.gpu_profiling
        )

        draft_config = (
            self.pipeline_config.draft_model_config.huggingface_config
        )

        self.vocab_size = (
            self.pipeline_config.model_config._huggingface_config.vocab_size
        )

        # Retrieve Encoding, and Files for Draft Model
        if self.pipeline_config.draft_model_config is None:
            raise ValueError(
                "draft_model must be provided for speculative decoding"
            )

        draft_hf_repo = (
            self.pipeline_config.draft_model_config.huggingface_weight_repo
        )
        encodings = draft_hf_repo.supported_encodings
        if not encodings:
            raise ValueError(
                "could not identify supported encodings for draft model."
            )

        if len(encodings) > 1:
            raise ValueError(
                "repos that only support one encoding, currently supported for draft model."
            )

        # Get weight files
        weight_files = draft_hf_repo.files_for_encoding(encoding=encodings[0])

        if not weight_files:
            raise ValueError("could not identify weight_files for draft model.")

        _draft_weights_format = list(weight_files.keys())[0]

        draft_weight_paths: list[Path] = []
        if (
            self.pipeline_config.draft_model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            draft_weight_paths = download_weight_files(
                huggingface_model_id=self.pipeline_config.draft_model_config.model_path,
                filenames=[str(x) for x in weight_files[_draft_weights_format]],
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

        self._draft_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=draft_session,
            huggingface_config=draft_config,
            encoding=encodings[0],
            devices=self.draft_devices,
            kv_cache_config=self.pipeline_config.draft_model_config.kv_cache_config,
            weights=draft_weights,
            adapter=weight_adapters.get(_draft_weights_format, None),
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
            msg = f"draft maximum sequence length ({draft_seq_len}) must match target maximum sequence length."
            raise ValueError(msg)

        self._ragged_token_merger = target_session.load(
            ragged_token_merger(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

    @traced
    def calculate_num_steps(
        self,
        model: PipelineModel,
        huggingface_config: AutoConfig,
        num_steps: int,
        context: T,
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
                f"Request {context.cache_seq_id} length ({context.current_length}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return (
            num_steps
            if num_available_steps > num_steps
            else num_available_steps
        )

    def _prepare_llama3_batch(
        self,
        model: PipelineModel,
        num_steps: int,
        return_n_logits: int,
        merged_draft_tokens: Tensor,
        merged_draft_offsets: Tensor,
        kv_cache_inputs: KVCacheInputsSequence,
    ) -> "Llama3Inputs":  # type: ignore
        from ..architectures.llama3.model import Llama3Inputs

        return Llama3Inputs(
            tokens=merged_draft_tokens,
            input_row_offsets=merged_draft_offsets,
            signal_buffers=[],
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )

    @traced
    def prepare_batch(
        self,
        model: PipelineModel,
        batch: list[T],
        num_steps: int,
        return_n_logits: int,
        is_draft: bool = False,
        merged_draft_tokens: Optional[Tensor] = None,
        merged_draft_offsets: Optional[Tensor] = None,
    ) -> tuple[ModelInputs, int]:
        # Claim cache rows
        for i, context in enumerate(batch):
            if not model.kv_manager.contains(context.cache_seq_id):
                model.kv_manager.external_claim([context.cache_seq_id])

            # Calculate num_steps.
            num_steps = self.calculate_num_steps(
                model, model.huggingface_config, num_steps, context, is_draft
            )

        kv_cache_inputs = model.kv_manager.fetch(
            cast(list[InputContext], batch), num_steps
        )
        if is_draft:
            return (
                model.prepare_initial_token_inputs(
                    context_batch=batch,
                    kv_cache_inputs=KVCacheInputsSequence(
                        kv_cache_inputs=kv_cache_inputs
                    ),
                    return_n_logits=return_n_logits,
                ),
                num_steps,
            )
        else:
            assert merged_draft_tokens is not None
            assert merged_draft_offsets is not None
            return (
                self._prepare_llama3_batch(
                    model,
                    num_steps,
                    return_n_logits,
                    merged_draft_tokens,
                    merged_draft_offsets,
                    kv_cache_inputs=KVCacheInputsSequence(
                        kv_cache_inputs=kv_cache_inputs
                    ),
                ),
                num_steps,
            )

    @traced
    def sample_draft_logits(
        self,
        batch: list[T],
        model_outputs: ModelOutputs,
        prev_tokens: Tensor,
        prev_logits: Tensor,
        top_k: Tensor,
        max_k: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        seed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        graph_inputs = [
            model_outputs.logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            seed,
            prev_logits,
        ]
        a, b, c = self._draft_sampler(*graph_inputs)[:3]
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        assert isinstance(c, Tensor)
        return (a, b, c)

    @traced
    def generate_draft_tokens(
        self, batch: list[T], num_steps: int
    ) -> tuple[int, Tensor, Tensor, ModelInputs, Tensor]:
        # Prepare the Batch
        model_inputs, num_steps = self.prepare_batch(
            self._draft_model,
            batch,
            num_steps,
            return_n_logits=1,
            is_draft=True,
        )

        # Create sampling parameters once for the entire batch
        top_k_np = np.array(
            [context.sampling_params.top_k for context in batch], dtype=np.int64
        )
        top_k = Tensor.from_numpy(top_k_np).to(self.draft_devices[0])
        max_k_np = np.array(np.max(top_k_np), dtype=np.int64)
        max_k = Tensor.from_numpy(max_k_np)
        temperature_np = np.array(
            [context.sampling_params.temperature for context in batch],
            dtype=np.float32,
        )
        temperature = Tensor.from_numpy(temperature_np).to(
            self.draft_devices[0]
        )
        top_p_np = np.array(
            [context.sampling_params.top_p for context in batch],
            dtype=np.float32,
        )
        top_p = Tensor.from_numpy(top_p_np).to(self.draft_devices[0])
        seed_np = np.array(
            [context.sampling_params.seed for context in batch], dtype=np.uint64
        )
        seed = Tensor.from_numpy(seed_np).to(self.draft_devices[0])

        # Generate tensor for generated tokens.
        generated_tokens = Tensor.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.draft_devices[0]
        )

        generated_logits = Tensor.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.draft_devices[0]
        )

        # Multi-step execution
        curr_step_inputs = model_inputs

        # num_steps first so that slice indexing is contiguous
        all_draft_logits = Tensor.zeros(
            (num_steps, len(batch), self.vocab_size),
            dtype=DType.float32,
            device=self.draft_devices[0],
        )

        for i in range(num_steps):
            # Execute the model and get next tokens.
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            all_draft_logits[i, :, :].inplace_copy_from(model_outputs.logits)

            # Sample next_token
            new_tokens, new_generated_tokens, new_generated_logits = (
                self.sample_draft_logits(
                    batch,
                    model_outputs,
                    generated_tokens,
                    generated_logits,
                    top_k,
                    max_k,
                    temperature,
                    top_p,
                    seed,
                )
            )
            generated_tokens = new_generated_tokens
            generated_logits = new_generated_logits

            # Increment cache lengths.
            assert isinstance(
                curr_step_inputs.kv_cache_inputs, KVCacheInputsSequence
            ), (
                "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
            )
            assert isinstance(
                curr_step_inputs.kv_cache_inputs.kv_cache_inputs, list
            ), "increment_cache_lengths instantiates and passes this as a list"
            curr_step_inputs.kv_cache_inputs.kv_cache_inputs = (
                self._draft_model.kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            # Prepare next token inputs.
            curr_step_inputs = self._draft_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )

        # The kv cache manager for the target model uses these indices to set the lengths of the cache. We bump them manually here even though the tokens array has not been filled. They are reset when doing the final update of the contexts after both draft and target models have run
        for i, context in enumerate(batch):
            context.bump_token_indices(active_idx=num_steps, end_idx=num_steps)
        return (
            num_steps,
            generated_tokens,
            generated_logits,
            model_inputs,
            all_draft_logits,
        )

    @traced
    def verify_draft_tokens_with_target_model(
        self,
        context_batch: list[T],
        num_draft_tokens_generated: int,
        draft_tokens: Tensor,
        draft_logits: Tensor,
        merged_draft_tokens: Tensor,
        merged_draft_offsets: Tensor,
        all_draft_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Prepare next token inputs for target model
        target_inputs, target_num_steps = self.prepare_batch(
            self._target_model,
            context_batch,
            # I believe, num steps in this scenario is 1, as we are only
            # generating one token beyond the draft tokens.
            num_steps=1,
            return_n_logits=num_draft_tokens_generated + 1,
            is_draft=False,
            merged_draft_tokens=merged_draft_tokens,
            merged_draft_offsets=merged_draft_offsets,
        )

        # Generate target tokens.
        target_outputs = self._target_model.execute(model_inputs=target_inputs)

        # Generate Final Samples
        assert target_outputs.logit_offsets is not None
        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self._rejection_sampler(
                draft_tokens,
                draft_logits,
                target_outputs.logits,
                target_outputs.logit_offsets,
                all_draft_logits,
            )
        )
        assert isinstance(first_rejected_tokens, Tensor)
        assert isinstance(recovered_tokens, Tensor)
        assert isinstance(bonus_tokens, Tensor)

        return first_rejected_tokens, recovered_tokens, bonus_tokens

    @traced
    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, execute both the draft model for num_steps and the target model for num_steps + 1 tokens, accepting final tokens via rejection sampling, returning the variable list of token integers."""

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())

        # Generate draft tokens.
        (
            num_draft_tokens_generated,
            draft_tokens,
            draft_logits,
            model_inputs,
            all_draft_logits,
        ) = self.generate_draft_tokens(context_batch, num_steps)

        # Merge draft tokens with target tokens
        merged_tokens, merged_offsets = self._ragged_token_merger(
            model_inputs.tokens,  # type: ignore
            model_inputs.input_row_offsets,  # type: ignore
            draft_tokens,
        )

        assert isinstance(merged_tokens, Tensor)
        assert isinstance(merged_offsets, Tensor)
        # Verify draft tokens with target model
        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self.verify_draft_tokens_with_target_model(
                context_batch,
                num_draft_tokens_generated,
                draft_tokens,
                draft_logits,
                merged_tokens,
                merged_offsets,
                all_draft_logits,
            )
        )

        self.update_contexts(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_tokens.to_numpy(),
            recovered_tokens=recovered_tokens.to_numpy(),
            bonus_tokens=bonus_tokens.to_numpy(),
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
        )

        res = self.build_response(batch=batch, context_batch=context_batch)

        # Maybe commit blocks into prefix cache
        self._target_model.kv_manager.step(
            cast(list[InputContext], context_batch)
        )
        self._draft_model.kv_manager.step(
            cast(list[InputContext], context_batch)
        )

        return res

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
        context_batch: list[T],
        first_rejected_tokens: np.ndarray,
        recovered_tokens: np.ndarray,
        bonus_tokens: np.ndarray,
        draft_tokens: np.ndarray,
        num_draft_tokens_generated: int,
    ) -> None:
        """Update contexts with the results of token generation.

        Args:
            context_batch: The list of context objects
            first_rejected_tokens: Array indicating the indices of first rejected tokens
            sampled_target_tokens: Array of sampled tokens from the target model
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
        self,
        batch: dict[str, T],
        context_batch: list[T],
    ) -> dict[str, TextGenerationResponse]:
        """Build response from updated contexts.

        Args:
            batch: The input batch dictionary mapping request IDs to contexts
            context_batch: The list of context objects

        Returns:
            Dictionary mapping request IDs to TextGenerationResponse objects
        """
        res: dict[str, TextGenerationResponse] = {}
        request_ids = list(batch.keys())

        for idx, context in enumerate(context_batch):
            res[request_ids[idx]] = TextGenerationResponse([], context.status)

            # Identify the Max Length
            context_max_length = upper_bounded_default(
                upper_bound=self._max_length, default=context.max_length
            )

            # TODO: This resets the context object for a new sequence.
            # We may want to make this more explicit
            for i, (token, log_probs) in enumerate(
                context.outstanding_completion_tokens()
            ):
                # Break early if beyond max length
                current_length = context.start_idx + 1

                if current_length >= context_max_length:
                    context.update_status(TextGenerationStatus.MAXIMUM_LENGTH)
                    res[request_ids[idx]].update_status(context.status)

                    res[request_ids[idx]].append_token(
                        TextResponse(token, log_probs)
                    )
                    break
                else:
                    res[request_ids[idx]].append_token(
                        TextResponse(token, log_probs)
                    )

        return res

    @traced
    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.

        """
        self._draft_model.kv_manager.release(context.cache_seq_id)
        self._target_model.kv_manager.release(context.cache_seq_id)
