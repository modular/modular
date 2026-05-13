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
"""MAX pipeline for model inference and generation (Text Generation variant)."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from abc import abstractmethod
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import numpy.typing as npt
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    load_devices,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    BatchLogitsProcessor,
    LogProbabilities,
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import (
    IncrementCacheLengthsProcessor,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.profiler import Tracer, traced
from max.support.algorithm import flatten2d

from .utils import (
    StructuredOutputHelper,
    calculate_num_steps,
    get_eos_tokens,
    update_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

from ..interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
)
from ..interfaces.generate import GenerateMixin
from ..sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")


@dataclasses.dataclass
class BatchInfo:
    """Information about a batch of requests passed to the pipeline."""

    past_seq_lens: list[int]
    """Coordinated list of past sequence lengths (i.e. context lengths)"""

    seq_lens: list[int]
    """Coordinated list of sequence lengths, i.e. prompt_len or 1"""

    num_steps: int
    """Number of steps to do in the pipeline"""


class TextGenerationPipelineInterface(
    Pipeline[
        TextGenerationInputs[TextGenerationContextType], TextGenerationOutput
    ],
    GenerateMixin[TextGenerationContextType, TextGenerationRequest],
    Generic[TextGenerationContextType],
):
    """Interface for text generation pipelines."""

    # TODO: Get rid of these fields
    _devices: list[Device]
    _pipeline_model: PipelineModelWithKVCache[TextGenerationContextType]

    @property
    @abstractmethod
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache managers for this pipeline."""
        ...


class TextGenerationPipeline(
    TextGenerationPipelineInterface[TextGenerationContextType],
    Generic[TextGenerationContextType],
):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TextGenerationContextType]],
        # TODO: This should be removed.
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextGenerationContextType,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
    ) -> None:
        """Initialize a text generation pipeline instance.

        This sets up devices, the inference session, tokenizer, KV-cache manager,
        sampling kernel, and loads model weights and adapters.

        Args:
            pipeline_config: Configuration for the pipeline and runtime behavior.
            pipeline_model: Concrete model implementation to use for execution.
            eos_token_id: Default EOS token id used when HF config does not supply
                one or to seed the EOS set.
            weight_adapters: Mapping from weights format to adapter implementation.
            tokenizer: Tokenizer implementation used to build contexts and decode.

        Raises:
            ValueError: If ``quantization_encoding`` is not configured in
                ``pipeline_config.model`` or if structured output is
                requested without a valid tokenizer delegate.
        """
        self._pipeline_config = pipeline_config
        model_config: MAXModelConfig = pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"Text generation pipeline requires a HuggingFace config for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        self._devices = load_devices(model_config.device_specs)
        self._tokenizer = tokenizer

        self.batch_info_output_fname = environ.get(
            "MAX_BATCH_INFO_FILENAME", None
        )
        self.batch_infos: list[BatchInfo] = []

        self._eos_token_id = get_eos_tokens(huggingface_config, eos_token_id)

        # Initialize structured output helper for constrained decoding.
        self._structured_output = StructuredOutputHelper.from_tokenizer(
            self.tokenizer,
            pipeline_config.sampling.enable_structured_output,
        )
        self.vocab_size = self._structured_output.vocab_size

        # Initialize Session.
        session = InferenceSession(devices=[*self._devices])
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = model_config.resolved_weight_paths()

        if not issubclass(pipeline_model, PipelineModelWithKVCache):
            raise ValueError(
                f"TextGenerationPipeline requires a model with KV cache support, found {pipeline_model.__name__}"
            )
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            kv_cache_config=model_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=ReturnLogits.ALL
            if self._pipeline_config.model.enable_echo
            else ReturnLogits.LAST_TOKEN,
        )

        available_cache_memory = model_config.kv_cache._available_cache_memory
        kv_params = self._pipeline_model.kv_params
        self._kv_manager = load_kv_manager(
            params=kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=self._pipeline_model.max_seq_len,
            session=session,
            available_cache_memory=available_cache_memory,
        )

        # Use the model's kv_params (not the manager's) because in
        # compile-only mode the manager is a Mock.
        if isinstance(kv_params, MultiKVCacheParams):
            # In case of multiple KV caches, multiple KV cache inputs are passed to `IncrementCacheLengthsProcessor.execute`, params only used for metadata.
            primary_params = kv_params.params[0]
        else:
            assert isinstance(kv_params, KVCacheParams)
            primary_params = kv_params
        self._increment_cache_lengths_processor = (
            IncrementCacheLengthsProcessor(
                session=session, params=primary_params
            )
        )

        # Load sampler.
        self._sampler_with_bitmask: Model | None = None
        self._sampler_without_bitmask: Model | None = None
        if pipeline_config.sampling.enable_structured_output:
            self._sampler_with_bitmask = session.load(
                token_sampler(
                    pipeline_config.sampling,
                    device=DeviceRef.from_device(self._devices[0]),
                )
            )
            cfg_without_bitmask = copy.deepcopy(pipeline_config.sampling)
            cfg_without_bitmask.enable_structured_output = False
            self._sampler_without_bitmask = session.load(
                token_sampler(
                    cfg_without_bitmask,
                    device=DeviceRef.from_device(self._devices[0]),
                )
            )
        else:
            self._sampler_without_bitmask = session.load(
                token_sampler(
                    pipeline_config.sampling,
                    device=DeviceRef.from_device(self._devices[0]),
                )
            )

        # Pre-allocate pinned buffer for D2H token copies only when structured
        # output is enabled. This buffer is used for async token transfers in
        # the guided decoding path. Allocated once and reused across batches.
        self._pinned_new_tokens: Buffer | None = None
        if (
            pipeline_config.sampling.enable_structured_output
            and not self._devices[0].is_host
        ):
            max_batch_size = pipeline_config.runtime.max_batch_size
            assert max_batch_size is not None, "max_batch_size must be set"
            self._pinned_new_tokens = DevicePinnedBuffer(
                shape=(max_batch_size,),
                dtype=DType.int64,
                device=self._devices[0],
            )

        self._identity_logit_offsets = (
            FusedSamplingProcessor.allocate_identity_logit_offsets(
                pipeline_config, self._devices[0]
            )
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Return the tokenizer used for building contexts and decoding."""
        return self._tokenizer

    def update_for_structured_output(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Update context and logits bitmask for structured output.

        If a ``json_schema`` is present and no matcher is set, this compiles a
        grammar matcher and installs it on the context. It may also jump ahead in
        generation and fills the per-request token bitmask used to constrain the
        next-token distribution.

        Args:
            context: Request context to update.
            bitmask: Optional preallocated bitmask buffer; updated in-place.
            index: Global position into the bitmask for this request.

        Raises:
            ValueError: If a JSON schema is provided but structured output is not
                enabled via sampling configuration.
        """
        self._structured_output.update_context(context, bitmask, index)

    def initialize_bitmask(
        self, batch: list[TextGenerationContextType]
    ) -> npt.NDArray[np.int32] | None:
        """Allocates a per-request token bitmask for structured decoding.

        Args:
            batch: The generation contexts for the batch.

        Returns:
            A bitmask array of shape [batch_size, vocab_size] if structured
            output is enabled; otherwise ``None``.
        """
        if not self._structured_output.enabled:
            return None

        if all(
            context.json_schema is None and context.grammar is None
            for context in batch
        ):
            return None

        return self._structured_output.allocate_bitmask(len(batch))

    @traced
    def prepare_batch(
        self,
        batches: list[list[TextGenerationContextType]],
        num_steps: int,
    ) -> tuple[
        Any,
        int,
        npt.NDArray[np.int32] | None,
        list[TextGenerationContextType],
    ]:
        """Prepare model inputs and ancillary state for multi-step execution.

        This flattens replica batches, optionally initializes constrained
        decoding bitmasks, ensures KV-cache reservations, clamps ``num_steps``
        per context, and builds initial model inputs.

        Args:
            batches: Per-replica list of contexts.
            num_steps: Desired number of steps to run.

        Returns:
            A tuple of:
                - ModelInputs: Prepared inputs for the first step.
                - int: The clamped number of steps to run.
                - Optional[np.ndarray]: The structured decoding bitmask or None.
                - list[TextGenerationContextType]: The flattened context batch.
        """
        replica_batches: list[list[TextGenerationContextType]] = [
            self._maybe_sort_loras(batch) for batch in batches
        ]
        flat_batch = flatten2d(replica_batches)

        # Initialize a bitmask for structured output.
        bitmask = self.initialize_bitmask(flat_batch)

        # Keep a global index for bitmask indexing.
        for i, context in enumerate(flat_batch):
            # Update state for structured output. Initialize a matcher if needed, this includes:
            # - Initializing a matcher if needed [once per request]
            # - Jumping ahead in generation if possible
            # - Updating the bitmask for the context.
            if bitmask is not None:
                self.update_for_structured_output(context, bitmask, i)

            # Update num_steps.
            num_steps = calculate_num_steps(
                context, num_steps, self._pipeline_model.max_seq_len
            )

        # Note: Multi-step execution with structured output is supported.
        # The bitmask is updated after each step in the
        # TextGenerationPipeline.execute loop.

        # Retrieve the KV Cache Inputs.
        kv_cache_inputs = self._kv_manager.runtime_inputs(
            replica_batches, num_steps
        )

        # Log batch details
        if self.batch_info_output_fname is not None:
            self._record_batch_info(flat_batch, num_steps)

        return (
            self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=replica_batches,
                kv_cache_inputs=kv_cache_inputs,
            ),
            num_steps,
            bitmask,
            flat_batch,
        )

    @traced
    def _maybe_sort_loras(
        self, batch: list[TextGenerationContextType]
    ) -> list[TextGenerationContextType]:
        """Optionally sorts the batch by LoRA IDs.

        Requests that use the same LoRA are placed adjacent to each other.
        """
        if self._pipeline_model._lora_manager is None:
            return batch

        return self._pipeline_model._lora_manager.sort_lora_batch(batch)

    def _update_bitmask_for_next_step(
        self,
        flat_batch: list[TextGenerationContextType],
        bitmask: npt.NDArray[np.int32],
        sampling_processor: FusedSamplingProcessor,
    ) -> None:
        """Update FSM state and bitmask for the next step in multi-step execution.

        After each token is sampled during multi-step execution with guided
        decoding, this method advances the FSM state for each context's matcher
        and recomputes the bitmask to reflect valid next tokens.

        Args:
            flat_batch: The batch of generation contexts.
            bitmask: The packed bitmask array to update in-place.
            sampling_processor: The sampling processor with the GPU bitmask
                and async token copy methods.
        """
        with Tracer("get_new_tokens"):
            # Wait for async D2H copy (started after sampling) and get tokens
            new_tokens_np = sampling_processor.get_new_tokens_numpy()

        for batch_idx, context in enumerate(flat_batch):
            if context.is_done or context.matcher is None:
                continue

            # Advance FSM with the sampled token (token buffer updated later)
            # new_tokens has shape (batch_size,) - 1D array
            token = int(new_tokens_np[batch_idx])
            with Tracer("advance_fsm"):
                if not context.advance_fsm(token):
                    raise RuntimeError(
                        f"FSM rejected token {token} during multi-step update. "
                        f"This indicates a mismatch between the bitmask and FSM state."
                    )

            # Fill the updated bitmask for this context
            with Tracer("fill_next_token_bitmask"):
                self._structured_output.fill_bitmask(
                    context, bitmask, batch_idx
                )

        with Tracer("sampling_processor_update_bitmask"):
            # Transfer updated bitmask to GPU
            sampling_processor.update_bitmask(bitmask)

    def _record_batch_info(self, contexts: Any, num_steps: int) -> None:
        """Record per-step batch statistics for diagnostics.

        Args:
            contexts: Contexts in the step, providing ``start_idx`` and
                ``active_length``.
            num_steps: Number of steps processed in this batch.

        Side Effects:
            Appends a ``BatchInfo`` entry to ``self.batch_infos``.
        """
        self.batch_infos.append(
            BatchInfo(
                past_seq_lens=[x.tokens.processed_length for x in contexts],
                seq_lens=[x.tokens.active_length for x in contexts],
                num_steps=num_steps,
            )
        )

    def __del__(self) -> None:
        """Flush recorded batch information to disk if configured.

        When ``MAX_BATCH_INFO_FILENAME`` is set, this writes a JSON file
        containing per-step batch statistics collected during execution.
        """
        if (
            hasattr(self, "batch_info_output_fname")
            and self.batch_info_output_fname is not None
        ):
            output = {
                "batch_data": [dataclasses.asdict(x) for x in self.batch_infos]
            }
            with open(self.batch_info_output_fname, "w") as f:
                json.dump(output, f, indent=2)
                f.flush()  # Refer to MAXSERV-893

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Processes the batch and returns decoded tokens.

        Given a batch, executes the graph for num_steps in a multi-step
        scenario, then decodes the tokens and returns the list of decoded
        tokens.
        """
        device0 = self._devices[0]
        pinned = not device0.is_host
        # Prepare the batch.
        model_inputs, num_steps, bitmask, flat_batch = self.prepare_batch(
            inputs.batches, inputs.num_steps
        )

        batch_processors: list[BatchLogitsProcessor] = []
        if len(flat_batch) > 0:
            # If structured output is present in the batch, use the sampler with bitmask.
            sampler: Model
            if bitmask is not None:
                assert self._sampler_with_bitmask is not None, (
                    "Sampler must be built with bitmask sampling"
                )
                sampler = self._sampler_with_bitmask
            else:
                assert self._sampler_without_bitmask is not None
                sampler = self._sampler_without_bitmask

            with Tracer("FusedSamplingProcessor"):
                sampling_processor = FusedSamplingProcessor(
                    sampler=sampler,
                    pipeline_config=self._pipeline_config,
                    context_batch=flat_batch,
                    num_steps=num_steps,
                    device=device0,
                    pinned_new_tokens=self._pinned_new_tokens,
                    identity_logit_offsets=self._identity_logit_offsets,
                    bitmask=bitmask,
                    vocab_size=self.vocab_size,
                )

            batch_processors.append(sampling_processor)

        curr_step_inputs = model_inputs
        batch_log_probabilities: list[list[LogProbabilities | None]] = []
        # Launch first forward pass before entering the loop.
        model_outputs = self._launch_forward_pass(
            curr_step_inputs, flat_batch, num_steps, step=0
        )
        for i in range(num_steps):
            # model_outputs is always valid here - either from initial launch
            # (i=0) or from pre-launch at end of previous iteration (i>0).

            # Validate output. This is more of an internal check that the model
            # is implemented correctly.
            if (
                self._pipeline_config.sampling.enable_variable_logits
                and model_outputs.logit_offsets is None
            ):
                raise ValueError(
                    "Model must return logit_offsets when enable_variable_logits is True."
                )

            # Continue and execute the next step if the batch.
            if len(flat_batch) == 0:
                continue

            # Sample next token.
            with Tracer("sample_next_token_step_{i}"):
                sample_logits, sample_offsets = (
                    sampling_processor.logits_for_sampling(
                        logits=model_outputs.logits,
                        next_token_logits=model_outputs.next_token_logits,
                        logit_offsets=model_outputs.logit_offsets,
                    )
                )
                apply_logits_processors(
                    context_batch=flat_batch,
                    batch_logits=sample_logits,
                    batch_logit_offsets=sample_offsets,
                    batch_processors=batch_processors,
                )
                new_tokens = sampling_processor.new_tokens
                assert new_tokens is not None

            # Start async D2H copy of tokens for FSM update (if needed).
            # This overlaps the transfer with log probs computation and other work.
            # Skip on last iteration since _update_bitmask_for_next_step won't be called.
            if bitmask is not None and i < num_steps - 1:
                with Tracer(f"start_async_token_copy_step_{i}"):
                    sampling_processor.start_async_token_copy()

            if inputs.enable_log_probs:
                with Tracer("compute_log_probabilities_step_{i}"):
                    try:
                        batch_log_probabilities.append(
                            self._pipeline_model.compute_log_probabilities(
                                self.session,
                                curr_step_inputs,
                                model_outputs,
                                new_tokens,
                                inputs.batch_top_log_probs,
                                inputs.batch_echo,
                            )
                        )
                    except NotImplementedError:
                        logger.warning(
                            "Unable to compute log probabilities for"
                            f" {self._pipeline_config.model.model_path}"
                        )
                        batch_log_probabilities.append(
                            [None for _ in flat_batch]
                        )

            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                break

            # Prepare inputs for next iteration before bitmask update.
            # This allows us to launch the next forward pass early.
            curr_step_inputs.kv_cache_inputs = (
                self._increment_cache_lengths_processor.execute(
                    curr_step_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            with Tracer(f"prepare_next_token_inputs_{i}"):
                curr_step_inputs = (
                    self._pipeline_model.prepare_next_token_inputs(
                        new_tokens, curr_step_inputs
                    )
                )

            # Launch next forward pass before bitmask update (if any).
            # For guided decoding, this overlaps GPU forward pass with CPU-side
            # FSM update and the cuStreamSynchronize wait in get_new_tokens_numpy().
            model_outputs = self._launch_forward_pass(
                curr_step_inputs, flat_batch, num_steps, step=i + 1
            )

            # Update FSM state and bitmask for next step (multi-step guided decoding).
            # This blocks on cuStreamSynchronize but GPU is busy with forward pass.
            if bitmask is not None:
                with Tracer(f"update_bitmask_step_{i}"):
                    self._update_bitmask_for_next_step(
                        flat_batch, bitmask, sampling_processor
                    )

        # Return early if the batch is empty.
        if len(flat_batch) == 0:
            return {}

        # Do the copy to host for each token generated.
        with Tracer("d2h_generated_tokens"):
            generated_tokens_device = sampling_processor.generated_tokens
            # Allocate a pinned tensor on the host for faster async d2h transfer
            # speeds. If the model is on host, then fall back to normal pageable
            # memory.
            # Note that we do not want to use `DevicePinnedBuffer` here.
            generated_tokens_host = Buffer(
                shape=generated_tokens_device.shape,
                dtype=generated_tokens_device.dtype,
                device=device0,
                pinned=pinned,
            )
            generated_tokens_host.inplace_copy_from(generated_tokens_device)
            # We assume that the call to `.to_numpy()` will insert a device
            # synchronize to guarantee that the async d2h transfer is done.
            # However, if this API changes we will have to add an explicit
            # device0.synchronize() here.
            generated_tokens_np = generated_tokens_host.to_numpy()

        # Update the context object.
        # During multi-step execution with guided decoding, the FSM was already
        # advanced for steps 0..num_steps-2 in _update_bitmask_for_next_step.
        # Only the last step needs FSM advancement here.
        fsm_already_advanced = (num_steps - 1) if bitmask is not None else 0
        res = update_context_and_prepare_responses(
            generated_tokens_np,
            flat_batch,
            num_steps,
            batch_log_probabilities=batch_log_probabilities,
            enable_log_probs=inputs.enable_log_probs,
            fsm_already_advanced_steps=fsm_already_advanced,
        )

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        self._kv_manager.step(inputs.batches)

        return res

    def _launch_forward_pass(
        self,
        curr_step_inputs: ModelInputs,
        flat_batch: list[TextGenerationContextType],
        num_steps: int,
        step: int,
    ) -> ModelOutputs:
        with Tracer(f"multistep_execution_loop_step_{step}"):
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=curr_step_inputs
                )
                return model_outputs
            except Exception:
                batch_size = len(flat_batch)
                cache_tokens = sum(
                    ctx.tokens.processed_length for ctx in flat_batch
                )
                input_tokens = sum(
                    ctx.tokens.active_length for ctx in flat_batch
                )
                logger.error(
                    "Encountered an exception while executing batch: "
                    f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}, {num_steps=:}"
                )
                raise  # re-raise the original exception

    def release(self, request_id: RequestID) -> None:
        """Release model-specific resources for a completed request.

        Primary and extra KV cache lifecycle is managed by the batch
        constructor.  This method handles model-specific cleanup only
        (e.g. vision encoder cache).
        """
        if hasattr(self._pipeline_model, "release"):
            self._pipeline_model.release(request_id)

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        return self._kv_manager
