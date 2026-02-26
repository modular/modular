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

import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher
from max.driver import Buffer, Device, load_devices
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
from max.kv_cache import PagedKVCacheManager, load_kv_manager
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputsSequence, KVCacheParams
from max.profiler import Tracer, traced
from max.support.algorithm import flatten2d
from transformers import PreTrainedTokenizerFast

from .utils import (
    calculate_num_steps,
    get_eos_tokens,
    get_weight_paths,
    update_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

from ..interfaces import PipelineModel, PipelineModelWithKVCache
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

        # Create a grammar compiler if constrained decoding is enabled
        self.vocab_size = None

        if pipeline_config.sampling.enable_structured_output:
            assert hasattr(self.tokenizer, "delegate")
            hf_tokenizer = self.tokenizer.delegate
            assert isinstance(hf_tokenizer, PreTrainedTokenizerFast)
            self.vocab_size = len(hf_tokenizer)
            self._tokenizer_info = llguidance.hf.from_tokenizer(
                hf_tokenizer, n_vocab=self.vocab_size
            )

        # Initialize Session.
        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = get_weight_paths(model_config)

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

        kv_params = self._pipeline_model.kv_params
        assert isinstance(kv_params, KVCacheParams)
        self._kv_manager: PagedKVCacheManager = load_kv_manager(
            params=kv_params,
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=self._pipeline_model.max_seq_len,
            session=session,
            available_cache_memory=model_config.kv_cache._available_cache_memory,
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
        if context.json_schema and context.matcher is None:
            if not self._pipeline_config.sampling.enable_structured_output:
                raise ValueError(
                    "json_schema provided but constrained decoding is not enabled."
                )

            try:
                serialized_grammar = LLMatcher.grammar_from_json_schema(
                    context.json_schema,
                )
                matcher = LLMatcher(self._tokenizer_info, serialized_grammar)
                context.set_matcher(matcher)
            except Exception as e:
                msg = f"Json schema provided in request cannot be compiled to valid grammar.                 Please update your json schema to produce valid structured output. From llguidance: {e}"
                logger.warning(msg)
                # I am removing the json_schema, so it doesn't try to load the grammar repeatedly.
                context.json_schema = None  # type: ignore

        if context.matcher:
            # Jump ahead in generation if possible.
            jump_forward_tokens = context.matcher.compute_ff_tokens()
            for token in jump_forward_tokens:
                context.jump_ahead(token)

            # Update the bitmask for the context.
            llguidance.numpy.fill_next_token_bitmask(
                context.matcher, bitmask, index=index
            )

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
        if not self._pipeline_config.sampling.enable_structured_output:
            return None

        if self.vocab_size is None:
            raise ValueError("vocab_size must be set to use structured output")

        if all(context.json_schema is None for context in batch):
            return None

        return llguidance.numpy.allocate_token_bitmask(
            len(batch), self.vocab_size
        )

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

        # If structured output is enabled for a specific request, we only need to run for a single step.
        # This is the only check to ensure that we do not apply an outdated bitmask to new inputs, during the next step.
        if bitmask is not None:
            num_steps = 1

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
                kv_cache_inputs=KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs
                ),
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

        replica_batch_sizes = [len(batch) for batch in inputs.batches]
        active_replica_idx = None
        if sum(replica_batch_sizes) == 1:
            active_replica_idx = next(
                (
                    replica_idx
                    for replica_idx, batch_size in enumerate(
                        replica_batch_sizes
                    )
                    if batch_size > 0
                ),
                None,
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
                    bitmask=bitmask,
                    vocab_size=self.vocab_size,
                )

            batch_processors.append(sampling_processor)

        curr_step_inputs = model_inputs
        batch_log_probabilities: list[list[LogProbabilities | None]] = []
        for i in range(num_steps):
            with Tracer(f"multistep_execution_loop_step_{i}"):
                # Execute the model and get next tokens.
                try:
                    model_outputs = self._pipeline_model.execute(
                        model_inputs=curr_step_inputs
                    )
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
                apply_logits_processors(
                    context_batch=flat_batch,
                    batch_logits=model_outputs.logits,
                    batch_logit_offsets=model_outputs.logit_offsets,
                    batch_processors=batch_processors,
                )
                new_tokens = sampling_processor.new_tokens
                assert new_tokens is not None

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

            assert isinstance(
                curr_step_inputs.kv_cache_inputs, KVCacheInputsSequence
            ), (
                "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
            )
            assert isinstance(
                curr_step_inputs.kv_cache_inputs.kv_cache_inputs, list
            ), "increment_cache_lengths instantiates and passes this as a list"
            curr_step_inputs.kv_cache_inputs.kv_cache_inputs = (
                self._kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )
            with Tracer(f"prepare_next_token_inputs_{i}"):
                curr_step_inputs = (
                    self._pipeline_model.prepare_next_token_inputs(
                        new_tokens, curr_step_inputs
                    )
                )

        # Return early if the batch is empty.
        if len(flat_batch) == 0:
            return {}

        # Do the copy to host for each token generated.
        with Tracer("D2H generated_tokens") as tracer:
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
        res = update_context_and_prepare_responses(
            generated_tokens_np,
            flat_batch,
            num_steps,
            batch_log_probabilities=batch_log_probabilities,
            enable_log_probs=inputs.enable_log_probs,
        )

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        self._kv_manager.step(inputs.batches)

        return res

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager.

        Note: KV cache lifecycle is now managed by the scheduler. This method
        is kept for interface compatibility but is a no-op for regular pipelines.
        """
        # KV cache release is handled by the scheduler via batch_constructor
        pass

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        return self._kv_manager
