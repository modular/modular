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
"""MAX pipeline for model inference and generation (Text Generation variant)."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import llguidance.hf
import llguidance.numpy
import numpy as np
import numpy.typing as npt
from llguidance import LLMatcher
from max.driver import load_devices
from max.engine import Model
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
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
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputsSequence
from max.profiler import Tracer, traced
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from ..config import PipelineConfig

from ..config_enums import RepoType
from ..hf_utils import download_weight_files
from ..interfaces import PipelineModel
from ..interfaces.generate import GenerateMixin
from ..sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")


@dataclasses.dataclass
class BatchInfo:
    """Information about a batch of requests passed to the pipeline"""

    past_seq_lens: list[int]
    """Coordinated list of past sequence lengths (i.e. context lengths)"""

    seq_lens: list[int]
    """Coordinated list of sequence lengths, i.e. prompt_len or 1"""

    num_steps: int
    """Number of steps to do in the pipeline"""


class TextGenerationPipeline(
    Pipeline[
        TextGenerationInputs[TextGenerationContextType], TextGenerationOutput
    ],
    GenerateMixin[TextGenerationContextType, TextGenerationRequest],
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
                ``pipeline_config.model_config`` or if structured output is
                requested without a valid tokenizer delegate.
        """
        self._pipeline_config = pipeline_config
        self._devices = load_devices(pipeline_config.model_config.device_specs)
        self._weight_adapters = weight_adapters
        self._tokenizer = tokenizer

        self.batch_info_output_fname = environ.get(
            "MAX_BATCH_INFO_FILENAME", None
        )
        self.batch_infos: list[BatchInfo] = []

        # Expand eos tokens if more are provided in pipeline_config
        if (
            "eos_token_id"
            in self._pipeline_config.model_config.huggingface_config
        ):
            eos_tokens = self._pipeline_config.model_config.huggingface_config.eos_token_id
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
                msg = f"eos_token_id in huggingface_config is neither int or list: {eos_tokens}"
                logger.warning(msg)
                self._eos_token_id = set([eos_token_id])

        else:
            self._eos_token_id = set([eos_token_id])

        # Create a grammar compiler if constrained decoding is enabled
        self.vocab_size = None

        if pipeline_config.sampling_config.enable_structured_output:
            assert hasattr(self.tokenizer, "delegate")
            hf_tokenizer = self.tokenizer.delegate
            assert isinstance(hf_tokenizer, PreTrainedTokenizerFast)
            self.vocab_size = len(hf_tokenizer)
            self._tokenizer_info = llguidance.hf.from_tokenizer(
                hf_tokenizer, n_vocab=self.vocab_size
            )

        # Initialize Session.
        from max.engine import InferenceSession  # local import to avoid cycles

        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not self._pipeline_config.model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weight id, if different than the model_path

        # TODO: These should ideally not call _weights_repo_id directly. I believe
        # huggingface_weight_repo_id property can be used here?
        weight_model_id = (
            self._pipeline_config.model_config._weights_repo_id
            if self._pipeline_config.model_config._weights_repo_id
            else self._pipeline_config.model_config.model_path
        )

        weight_paths: list[Path] = []
        if (
            self._pipeline_config.model_config.huggingface_weight_repo.repo_type
            == RepoType.online
        ):
            # Download weight files if not existent.
            weight_paths = download_weight_files(
                huggingface_model_id=weight_model_id,
                filenames=[
                    str(x)
                    for x in self._pipeline_config.model_config.weight_path
                ],
                revision=self._pipeline_config.model_config.huggingface_weight_revision,
                force_download=self._pipeline_config.model_config.force_download,
            )
        else:
            # Make sure the weight paths are absolute paths
            weight_paths = [
                self._pipeline_config.model_config.model_path / x
                for x in self._pipeline_config.model_config.weight_path
            ]

        # late imports to minimize header deps
        from max.graph.weights import (
            load_weights as _load_weights,
        )
        from max.graph.weights import (
            weights_format as _weights_format,
        )

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self._pipeline_config.model_config.huggingface_config,
            encoding=self._pipeline_config.model_config.quantization_encoding,
            devices=self._devices,
            kv_cache_config=self._pipeline_config.model_config.kv_cache_config,
            weights=_load_weights(weight_paths),
            adapter=self._weight_adapters.get(
                _weights_format(weight_paths), None
            ),
            return_logits=ReturnLogits.ALL
            if self._pipeline_config.enable_echo
            else ReturnLogits.LAST_TOKEN,
        )

        # Load sampler.
        from max.graph import DeviceRef as _DeviceRef

        self._sampler_with_bitmask: Model | None = None
        if self._pipeline_config.sampling_config.enable_structured_output:
            self._sampler_with_bitmask = session.load(
                token_sampler(
                    self._pipeline_config.sampling_config,
                    device=_DeviceRef.from_device(self._devices[0]),
                )
            )
            cfg_without_bitmask = copy.deepcopy(
                self._pipeline_config.sampling_config
            )
            cfg_without_bitmask.enable_structured_output = False
            self._sampler_without_bitmask = session.load(
                token_sampler(
                    cfg_without_bitmask,
                    device=_DeviceRef.from_device(self._devices[0]),
                )
            )
        else:
            self._sampler_without_bitmask = session.load(
                token_sampler(
                    self._pipeline_config.sampling_config,
                    device=_DeviceRef.from_device(self._devices[0]),
                )
            )
            self._sampler_with_bitmask = None

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

    @property
    def kv_managers(
        self,
    ) -> list[Any]:
        """Return the list of KV cache managers backing this pipeline."""
        return [self._pipeline_model.kv_manager]

    def calculate_num_steps(
        self,
        num_steps: int,
        context: TextGenerationContextType,
    ) -> int:
        """Compute the number of generation steps allowed for a context.

        The value is clamped by the remaining capacity with respect to
        the model's configured ``max_seq_len``.

        Args:
            num_steps: Desired number of steps to attempt.
            context: The context whose sequence length constraints apply.

        Returns:
            The number of steps to execute for this context (>= 1).

        Raises:
            ValueError: If the current request length is already >= ``max_seq_len``.
        """
        max_seq_len = self._pipeline_model.max_seq_len
        num_available_steps = context.compute_num_available_steps(max_seq_len)

        if num_available_steps <= 0:
            raise ValueError(
                f"Request {context.request_id} length ({context.current_length}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return min(num_available_steps, num_steps)

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
            if not self._pipeline_config.sampling_config.enable_structured_output:
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
        """Allocate a per-request token bitmask for structured decoding.

        Args:
            batch_size: Number of requests in the batch.

        Returns:
            A bitmask array of shape [batch_size, vocab_size] if structured
            output is enabled; otherwise ``None``.
        """
        if not self._pipeline_config.sampling_config.enable_structured_output:
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
        batches: list[dict[RequestID, TextGenerationContextType]],
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
            batches: Per-replica mapping of ``RequestID`` to context.
            num_steps: Desired number of steps to run.

        Returns:
            A tuple of:
                - ModelInputs: Prepared inputs for the first step.
                - int: The clamped number of steps to run.
                - Optional[np.ndarray]: The structured decoding bitmask or None.
                - list[TextGenerationContextType]: The flattened context batch.
        """
        # Initialize a flat batch of contexts and their replica ids.
        replica_ids: list[int] = [
            replica_idx
            for replica_idx, batch in enumerate(batches)
            for _ in batch.values()
        ]
        flat_batch: list[TextGenerationContextType] = [
            context
            for batch in batches
            for context in self._maybe_sort_loras(batch).values()
        ]

        # Initialize a bitmask for structured output.
        bitmask = self.initialize_bitmask(flat_batch)

        # Keep a global index for bitmask indexing.
        i = 0
        for i, (replica_idx, context) in enumerate(
            zip(replica_ids, flat_batch, strict=False)
        ):
            # Update state for structured output. Initialize a matcher if needed, this includes:
            # - Initializing a matcher if needed [once per request]
            # - Jumping ahead in generation if possible
            # - Updating the bitmask for the context.
            if bitmask is not None:
                self.update_for_structured_output(context, bitmask, i)

            if not self._pipeline_model.kv_manager.contains(context.request_id):
                self._pipeline_model.kv_manager.external_claim(
                    context.request_id, replica_idx=replica_idx
                )

            # Update num_steps.
            num_steps = self.calculate_num_steps(num_steps, context)

        # If structured output is enabled for a specific request, we only need to run for a single step.
        # This is the only check to ensure that we do not apply an outdated bitmask to new inputs, during the next step.
        if bitmask is not None:
            num_steps = 1

        # Retrieve the KV Cache Inputs.
        kv_cache_inputs = self._pipeline_model.kv_manager.fetch(
            flat_batch, num_steps
        )

        # Log batch details
        if self.batch_info_output_fname is not None:
            self._record_batch_info(flat_batch, num_steps)

        return (
            self._pipeline_model.prepare_initial_token_inputs(
                context_batch=flat_batch,
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
        self, batch: dict[RequestID, TextGenerationContextType]
    ) -> dict[RequestID, TextGenerationContextType]:
        """
        Maybe sorts the batch by LoRA Ids. Requests that use the same LoRA need
        to be adjacent to each other.
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
                past_seq_lens=[x.start_idx for x in contexts],
                seq_lens=[x.active_length for x in contexts],
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
    def update_context_and_prepare_responses(
        self,
        generated_tokens_host: npt.NDArray[np.int32],
        batch_log_probabilities: list[list[LogProbabilities | None]],
        flat_batch: list[TextGenerationContextType],
        num_steps: int,
        enable_log_probs: bool,
    ) -> dict[RequestID, TextGenerationOutput]:
        """
        Update the context objects and prepare the response objects for each context in the batch after generation.

        Args:
            generated_tokens_host: Array of generated tokens on the host, indexed as [batch, step].
            batch_log_probabilities: List of per-step log probability outputs (or None), each entry is a list per batch for that step.
            flat_batch: List of generation contexts, one per request, matching batch dimension.
            num_steps: Number of generation steps to process for each context.
            enable_log_probs: Whether to include log probability data in outputs.

        Returns:
            A dictionary mapping request IDs to their respective generation outputs.
        """
        res: dict[RequestID, TextGenerationOutput] = {}
        for batch_index, context in enumerate(flat_batch):
            for step in range(num_steps):
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                # Get Log probs if needed.
                log_probs: LogProbabilities | None = None
                if enable_log_probs and step < len(batch_log_probabilities):
                    log_probs_for_step = batch_log_probabilities[step]
                    if log_probs_for_step and batch_index < len(
                        log_probs_for_step
                    ):
                        log_probs = log_probs_for_step[batch_index]

                context.update(
                    new_token=next_token, log_probabilities=log_probs
                )

                if context.is_done:
                    break

            res[context.request_id] = context.to_generation_output()

        return res

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
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
                sampler = self._sampler_without_bitmask

            sampling_processor = FusedSamplingProcessor(
                sampler=sampler,
                pipeline_config=self._pipeline_config,
                context_batch=flat_batch,
                num_steps=num_steps,
                device=self._devices[0],
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
                    cache_tokens = sum(ctx.start_idx for ctx in flat_batch)
                    input_tokens = sum(ctx.active_length for ctx in flat_batch)
                    logger.error(
                        "Encountered an exception while executing batch: "
                        f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}, {num_steps=:}"
                    )
                    raise  # re-raise the original exception

            # Validate output. This is more of an internal check that the model
            # is implemented correctly.
            if (
                self._pipeline_config.sampling_config.enable_variable_logits
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
                            f" {self._pipeline_config.model_config.model_path}"
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
                self._pipeline_model.kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )

        # Return early if the batch is empty.
        if len(flat_batch) == 0:
            return {}

        # Do the copy to host for each token generated.
        with Tracer("generated_tokens.to(CPU())") as tracer:
            generated_tokens_host = (
                sampling_processor.generated_tokens.to_numpy()
            )

        # Update the context object.
        res = self.update_context_and_prepare_responses(
            generated_tokens_host,
            batch_log_probabilities,
            flat_batch,
            num_steps,
            inputs.enable_log_probs,
        )

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        self._pipeline_model.kv_manager.step(flat_batch)

        return res

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager."""
        self._pipeline_model.kv_manager.release(request_id)
