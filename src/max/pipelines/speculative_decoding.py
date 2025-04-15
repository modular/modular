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
from typing import Any, TypeVar, cast

import numpy as np
from max.driver import Tensor, load_devices, scan_available_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputsSequence
from transformers import AutoConfig

from .core import (
    InputContext,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
)
from .hf_utils import download_weight_files
from .pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    upper_bounded_default,
)
from .sampling import rejection_sampler, token_sampler

T = TypeVar("T", bound=InputContext)

logger = logging.getLogger("max.pipelines")


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

        weight_paths = download_weight_files(
            huggingface_model_id=target_hf_repo.repo_id,
            filenames=[
                str(x) for x in self.pipeline_config.model_config.weight_path
            ],
            revision=self.pipeline_config.model_config.huggingface_weight_revision,
            max_workers=8,
        )
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

        # Calculate Max Length
        self._max_length = self._target_model.calculate_max_seq_len(
            self.pipeline_config,
            huggingface_config=self.pipeline_config.model_config.huggingface_config,
        )

        # Load draft model
        # For now, we are assuming we are placing the draft model will sit
        self.draft_devices = load_devices(scan_available_devices()[:1])
        draft_session = InferenceSession(devices=self.draft_devices)

        draft_config = (
            self.pipeline_config.draft_model_config.huggingface_config
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
        weight_files = draft_hf_repo.files_for_encoding(
            encoding=encodings[0],
        )

        if not weight_files:
            raise ValueError("could not identify weight_files for draft model.")

        _draft_weights_format = list(weight_files.keys())[0]
        _draft_weight_paths = download_weight_files(
            huggingface_model_id=self.pipeline_config.draft_model_config.model_path,
            filenames=[str(x) for x in weight_files[_draft_weights_format]],
            revision=None,
            max_workers=8,
        )
        draft_weights = load_weights(_draft_weight_paths)

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
            token_sampler(draft_sampling_config, return_logits=True)
        )

        # Load rejection sampler
        self._rejection_sampler = target_session.load(
            rejection_sampler(self.pipeline_config.sampling_config)
        )

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

    def calculate_num_steps(
        self,
        model: PipelineModel,
        huggingface_config: AutoConfig,
        num_steps: int,
        context: T,
    ) -> int:
        max_seq_len = model.calculate_max_seq_len(
            self.pipeline_config,
            huggingface_config=huggingface_config,
        )
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

    def prepare_batch(
        self,
        model: PipelineModel,
        batch: list[T],
        num_steps: int,
        return_n_logits: int,
    ) -> tuple[ModelInputs, int]:
        # Claim cache rows
        for i, context in enumerate(batch):
            if not model.kv_manager.contains(context.cache_seq_id):
                model.kv_manager.external_claim([context.cache_seq_id])

            # Calculate num_steps.
            num_steps = self.calculate_num_steps(
                model, model.huggingface_config, num_steps, context
            )

        kv_cache_inputs = model.kv_manager.fetch(
            cast(list[InputContext], batch), num_steps
        )

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

    def sample_draft_logits(
        self,
        model_outputs: ModelOutputs,
        prev_tokens: Tensor,
        prev_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        graph_inputs = [model_outputs.logits, prev_tokens, prev_logits]

        if model_outputs.logit_offsets:
            graph_inputs.append(model_outputs.logit_offsets)

        a, b, c = self._draft_sampler(*graph_inputs)[:3]
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        assert isinstance(c, Tensor)
        return (a, b, c)

    def generate_draft_tokens(
        self, batch: list[T], num_steps: int
    ) -> tuple[int, np.ndarray, Tensor]:
        # Prepare the Batch
        model_inputs, num_steps = self.prepare_batch(
            self._draft_model, batch, num_steps, return_n_logits=1
        )

        # Generate tensor for generated tokens.
        generated_tokens = Tensor.zeros(
            (len(batch), 0),
            dtype=DType.int64,
            device=self.draft_devices[0],
        )

        generated_logits = Tensor.zeros(
            (len(batch), 0),
            dtype=DType.float32,
            device=self.draft_devices[0],
        )

        # Multi-step execution
        curr_step_inputs = model_inputs
        for i in range(num_steps):
            # Execute the model and get next tokens.
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs,
            )

            # Sample next_token
            new_tokens, new_generated_tokens, new_generated_logits = (
                self.sample_draft_logits(
                    model_outputs,
                    generated_tokens,
                    generated_logits,
                )
            )
            generated_tokens = new_generated_tokens
            generated_logits = new_generated_logits

            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                break

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

        # TODO: E2EOPT-129
        # Copy to HOST
        generated_tokens_host = generated_tokens.to_numpy()

        # Ignore EOS, and update the Context objects via Jump Ahead
        # This will require us to manage the context element pointers
        # manually, at the end once we've accepted/rejected the
        # necessary tokens.
        for i, context in enumerate(batch):
            for step in range(num_steps):
                new_token = int(generated_tokens_host[i, step])
                if context.ignore_eos:
                    is_eos = False
                elif new_token in self._eos_token_id:
                    is_eos = True
                else:
                    is_eos = False

                context.jump_ahead(
                    new_token=int(generated_tokens_host[i, step]),
                    is_eos=is_eos,
                )

        return num_steps, generated_tokens_host, generated_logits

    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, execute both the draft model for num_steps and the target model for num_steps + 1 tokens, accepting final tokens via rejection sampling, returning the variable list of token integers."""

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())

        # This is a bit of a hack, we should work out a better API for this.
        # The draft offset tracks how far behind the draft model should be.
        # We should bump the start_idx back up by this amount, to get back
        # to where the target model should start.
        for context in context_batch:
            context.bump_token_indices(start_idx=+context._draft_offset)  # type: ignore

        # Generate draft tokens.
        # This updates the context_batch object in place.
        num_draft_tokens_generated, draft_tokens, draft_logits = (
            self.generate_draft_tokens(context_batch, num_steps)
        )

        for context in context_batch:
            context.bump_token_indices(start_idx=-context._draft_offset)  # type: ignore

        # Prepare next token inputs for target model
        target_inputs, target_num_steps = self.prepare_batch(
            self._target_model,
            context_batch,
            # I believe, num steps in this scenario is 1, as we are only
            # generating one token beyond the draft tokens.
            num_steps=1,
            return_n_logits=num_draft_tokens_generated + 1,
        )

        # Generate target tokens.
        target_outputs = self._target_model.execute(
            model_inputs=target_inputs,
        )

        # Generate Final Samples
        assert target_outputs.logit_offsets is not None
        first_rejected_tokens, sampled_target_tokens = self._rejection_sampler(
            Tensor.from_numpy(draft_tokens).to(self.target_devices[0]),
            draft_logits,
            target_outputs.logits,
            target_outputs.logit_offsets,
        )

        assert isinstance(first_rejected_tokens, Tensor)
        assert isinstance(sampled_target_tokens, Tensor)
        sampled_target_tokens_host = sampled_target_tokens.to_numpy()
        res: dict[str, TextGenerationResponse] = {}
        request_ids = list(batch.keys())
        for idx, rejected_token_idx in enumerate(
            first_rejected_tokens.to_numpy()[0]
        ):
            context = context_batch[idx]

            res[request_ids[idx]] = TextGenerationResponse(
                [],
                context.active_status,  # type: ignore
            )

            context = context_batch[idx]
            rollback_count = num_draft_tokens_generated - rejected_token_idx

            # Check if the new token is EOS
            new_token = sampled_target_tokens_host[0][idx]
            if context.ignore_eos:
                is_eos = False
            elif new_token in self._eos_token_id:
                is_eos = True
            else:
                is_eos = False

            # If we are rolling back
            if rollback_count > 0:
                # This should return the start_idx by rollback_count
                context.rollback(rollback_count)

                # Update the context with the new target token.
                context.update(new_token, is_eos=is_eos)

                # Reset Draft Offset to 0
                context.set_draft_offset(idx=0)

            # If we are not rolling back
            else:
                # Update the new token
                context.update(new_token, is_eos=is_eos)

                # Bump the start_idx back by 1
                # This ensures the draft model picks up the last item in the sequence.
                context.set_draft_offset(idx=-1)

            res[request_ids[idx]].update_status(context.active_status)  # type: ignore

            # Identify the Max Length
            context_max_length = upper_bounded_default(
                upper_bound=self._max_length,
                default=context.max_length,
            )

            # TODO: This resets the context object for a new sequence.
            # We may want to make this more explicit
            for i, (token, log_probs) in enumerate(
                context.outstanding_completion_tokens()
            ):
                # Break early if beyond max length
                current_length = context.start_idx + 1

                if current_length >= context_max_length:
                    context.active_status = TextGenerationStatus.MAXIMUM_LENGTH  # type: ignore
                    res[request_ids[idx]].update_status(context.active_status)  # type: ignore

                    res[request_ids[idx]].append_token(
                        TextResponse(token, log_probs)
                    )
                    break
                else:
                    res[request_ids[idx]].append_token(
                        TextResponse(token, log_probs)
                    )

        # Maybe commit blocks into prefix cache
        self._target_model.kv_manager.step(
            cast(list[InputContext], context_batch)
        )
        self._draft_model.kv_manager.step(
            cast(list[InputContext], context_batch)
        )

        return res

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.

        """
        self._draft_model.kv_manager.release(context.cache_seq_id)
        self._target_model.kv_manager.release(context.cache_seq_id)
