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
"""Hugging Face Token Generation Pipeline."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

import torch
from max.driver import Device, Tensor, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import (
    Weights,
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    infer_optimal_batch_size,
)
from max.profiler import Tracer, traced
from transformers import AutoConfig, AutoTokenizer

if TYPE_CHECKING:
    from .config import PipelineConfig

from .config_enums import SupportedEncoding
from .context import InputContext
from .hf_utils import download_weight_files
from .interfaces import (
    LogProbabilities,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
)
from .kv_cache import KVCacheManager, KVCacheParams
from .max_config import KVCacheConfig
from .sampling import token_sampler

try:
    import xgrammar as xgr

    # This retrieves the last logger handler added
    # which presumably is the one initialized in xgrammar
    # and removes it, this stops our server logging from
    # doubling up.
    logger = logging.getLogger()
    handler = logger.handlers[-1]
    logger.removeHandler(handler)
except ImportError:
    pass

logger = logging.getLogger("max.pipelines")

ARCH_SAFE_VRAM_USAGE_LIMIT = {
    "DeepseekCoder": 0.96,
    "ExaoneForCausalLM": 0.96,
    "LlamaForCausalLM": 0.96,
    "MistralForCausalLM": 0.96,
}


def upper_bounded_default(upper_bound: int, default: int | None) -> int:
    """
    Given an upper bound and an optional default value, returns a final value
    that cannot exceed the upper bound.

    Args:
        default: The default value to use, or None to use the upper bound.
        upper_bound: The upper bound to use.

    Raises:
        ValueError: If the provided default value exceeds the upper bound.

    Returns:
        The final value.
    """
    if default is None:
        return upper_bound
    elif default > upper_bound:
        raise ValueError(
            f"default value provided ({default}) exceeds the upper bound ({upper_bound})"
        )
    return default


class ModelInputs:
    """
    Base class for model inputs.
    Use this class to encapsulate inputs for your model.
    You may store any number of dataclass fields

    Example:
        >>> class ReplitInputs(ModelInputs):
        ...     tokens: Tensor
        ...     input_row_offsets: Tensor
        ...
        ...     def __init__(self, tokens: Tensor, input_row_offsets: Tensor):
        ...         self.tokens = tokens
        ...         self.input_row_offsets = input_row_offsets
        ...
        >>> # Create tensors
        >>> tokens = Tensor.zeros((1, 2, 3), DType.int64)
        >>> input_row_offsets = Tensor.zeros((1, 1, 1), DType.int64)
        >>> # Initialize inputs
        >>> inputs = ReplitInputs(tokens=tokens, input_row_offsets=input_row_offsets)
        >>> # Access tensors
        >>> list(inputs) == [tokens, input_row_offsets]
        True
    """

    kv_cache_inputs: KVCacheInputs | None = None


@dataclass(frozen=True)
class ModelOutputs:
    logits: Tensor
    """Logits for a variable number of tokens per sequence."""

    next_token_logits: Tensor | None = None
    """Logits for just the next token."""

    logit_offsets: Tensor | None = None
    """Offsets to access variable length logits for each sequence."""


T = TypeVar("T", bound=InputContext)


class PipelineModel(ABC, Generic[T]):
    """A pipeline model with setup, input preparation and execution methods."""

    _MAX_DEFAULT_BATCH_SIZE = 4096
    _MIN_DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter],
        return_n_logits: int,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.huggingface_config = huggingface_config
        self.encoding = encoding
        self.devices = devices
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_n_logits = return_n_logits

        if isinstance(self, KVCacheMixin):
            self.kv_manager = self.load_kv_manager(
                session, self.kv_cache_config._available_cache_memory
            )

    @property
    def dtype(self) -> DType:
        return self.encoding.dtype

    @classmethod
    @abstractmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate the optimal max sequence length for the model.
        Models are expected to implement this method.

        Example:
            >>> class MistralModel(PipelineModel):
            ...     @classmethod
            ...     def calculate_max_seq_len(cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig) -> int:
            ...         try:
            ...             return upper_bounded_default(
            ...                 upper_bound=huggingface_config.max_seq_len,
            ...                 default=pipeline_config.max_length,
            ...             )
            ...         except ValueError as e:
            ...             msg = (
            ...                 "Unable to infer max_length for Mistral, the provided "
            ...                 f"max_length ({pipeline_config.max_length}) exceeds the "
            ...                 f"model's max_seq_len "
            ...                 f"({huggingface_config.max_seq_len})."
            ...             )
            ...             raise ValueError(msg) from e
            ...
        """
        raise NotImplementedError(
            "PipelineModel must implement calculate_max_seq_len"
        )

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Returns the KV cache params for the pipeline model."""
        ...

    # TODO(AITLIB-265): Remove this altogether from all PipelineModels.
    @classmethod
    @abstractmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Returns the number of layers for the pipeline model."""
        ...

    @classmethod
    def infer_optimal_batch_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        huggingface_config: AutoConfig,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Returns the estimated optimal batch size to run the model
        given current memory constraints."""
        if not issubclass(cls, KVCacheMixin):
            # we rely on the KVCache setup to know optimal batch size.
            # If we don't have that, default to BS=1.
            return 1
        elif len(devices) == 1 and devices[0].is_host:
            # batching on CPU is generally not useful, so we hard-code a batch size of 1.
            return 1

        # TODO we should map HF configs to a unified MAX Config object
        # this would help avoid these excessive calls to class methods.
        n_layers = cls.get_num_layers(
            huggingface_config=huggingface_config,
        )

        kv_params = cls.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=len(devices),
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        inferred_batch_size = infer_optimal_batch_size(
            params=kv_params,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=n_layers,
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

        # clamp the floor of the inferred batch size to 1 and the ceiling to 4096
        inferred_batch_size = max(
            cls._MIN_DEFAULT_BATCH_SIZE,
            min(inferred_batch_size, cls._MAX_DEFAULT_BATCH_SIZE),
        )
        return inferred_batch_size

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of our model."""

        # TODO move this logic to the PipelineModel instead of PipelineConfig class.
        # Better yet, make this more accurate by loading and measuring memory consumption
        # after we load the model
        return pipeline_config.model_config.weights_size()

    @abstractmethod
    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Executes the graph with the given inputs.

        Args:
            model_inputs: The model inputs to execute, containing tensors and any other
                required data for model execution.

        Returns:
            ModelOutputs containing the pipeline's output tensors.

        This is an abstract method that must be implemented by concrete PipelineModels
        to define their specific execution logic.
        """

    @abstractmethod
    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[T],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to `.execute()`.

        The inputs and functionality of this method can vary per model.
        For example, the model inputs could include:
        - Encoded tensors
        - A unique IDs for each tensor if this model uses a KV Cache manager.
        - kv_cache_inputs: The kv cache inputs required for the model. This
        should be None if the model does not use KV Cache.
        This function would batch the encoded tensors, claim a slot in the kv
        cache if the ID hasn't been seen before, and return the inputs and
        caches as a list of tensors."""
        ...

    @abstractmethod
    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        """Prepares the secondary inputs to be passed to `.execute()`.

        While `prepare_initial_token_inputs` is responsible for managing the initial inputs.
        This function is responsible for updating the inputs, for each step in a multi-step execution pattern.
        """
        ...

    def compute_log_probabilities(
        self,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        """Optional method that can be overridden to compute log probabilities.

        Args:
            model_inputs: Inputs to the model returned by
                `prepare_*_token_inputs()`.
            model_outputs: Outputs returned by `execute()`.
            next_tokens: Sampled tokens. Should have shape=[batch size]
            batch_top_n: Number of top log probabilities to return per input in
                the batch. For any element where `top_n == 0`, the
                LogProbabilities is skipped.
            batch_echo: Whether to include input tokens in the returned log
                probabilities.

        Returns:
            List of log probabilities.
        """
        raise NotImplementedError(
            f"Log probabilities not implemented for {type(self)}."
        )


@runtime_checkable
class KVCacheMixin(Protocol):
    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: Optional[int],
    ) -> KVCacheManager:
        """Provided a PipelineConfig and InferenceSession, loads the KV manager.

        Args:
            session: Inference session to compile and init the KV cache.
            available_cache_memory: Amount of memory available to the KV cache,
                in bytes.

        Returns:
            Either a single KV cache manager or a tuple of KV cache managers:
            one per input modality.
        """
        ...

    @classmethod
    @abstractmethod
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
        ...


class TextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: Type[PipelineModel],
        # TODO: This should be removed.
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
    ) -> None:
        self._pipeline_config = pipeline_config
        self._huggingface_config: Optional[AutoConfig] = None
        self._devices = load_devices(pipeline_config.model_config.device_specs)
        self._weight_adapters = weight_adapters

        # Expand eos tokens if more are provided in pipeline_config
        if self._pipeline_config.ignore_eos:
            self._eos_token_id = set([])
        elif "eos_token_id" in self.huggingface_config:
            eos_tokens = self.huggingface_config.eos_token_id
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

        # Create a grammar compiler if constrained decoding is enabled
        self.vocab_size = None
        if pipeline_config.sampling_config.enable_structured_output:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pipeline_config.model_config.model_path
            )
            self.vocab_size = len(self.tokenizer)
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )

            self._grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

        # Initialize Session.
        session = InferenceSession(devices=self._devices)

        # Enable profiling if enabled.
        session.gpu_profiling(
            self._pipeline_config.profiling_config.gpu_profiling
        )

        # Use experimental kernels if enabled by env var `USE_EXPERIMENTAL_KERNELS`.
        session._use_experimental_kernels(
            self._pipeline_config.use_experimental_kernels
        )

        # Load model.
        if not self._pipeline_config.model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weight id, if different than the model_path
        weight_model_id = (
            self._pipeline_config.model_config._weights_repo_id
            if self._pipeline_config.model_config._weights_repo_id
            else self._pipeline_config.model_config.model_path
        )

        # Download weight files if not existent.
        weight_paths = download_weight_files(
            huggingface_model_id=weight_model_id,
            filenames=[
                str(x) for x in self._pipeline_config.model_config.weight_path
            ],
            revision=self._pipeline_config.model_config.huggingface_revision,
            force_download=self._pipeline_config.model_config.force_download,
        )

        # Load weights
        weights = load_weights(weight_paths)
        _weight_format = weights_format(weight_paths)

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=self.huggingface_config,
            encoding=self._pipeline_config.model_config.quantization_encoding,
            devices=self._devices,
            kv_cache_config=self._pipeline_config.model_config.kv_cache_config,
            weights=weights,
            adapter=self._weight_adapters.get(_weight_format, None),
            return_n_logits=-1 if self._pipeline_config.enable_echo else 1,
        )

        # Load sampler.
        self._sampler = session.load(
            token_sampler(self._pipeline_config.sampling_config),
        )

    @property
    def huggingface_config(self) -> AutoConfig:
        if not self._huggingface_config:
            self._huggingface_config = AutoConfig.from_pretrained(
                self._pipeline_config.model_config.model_path,
                trust_remote_code=self._pipeline_config.model_config.trust_remote_code,
            )

        return self._huggingface_config

    def calculate_num_steps(
        self,
        num_steps: int,
        context: T,
    ) -> int:
        max_seq_len = self._pipeline_model.calculate_max_seq_len(
            self._pipeline_config,
            huggingface_config=self.huggingface_config,
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

    @traced
    def prepare_batch(
        self,
        batch: list[T],
        num_steps: int,
    ) -> tuple[ModelInputs, int, Optional[torch.Tensor]]:
        tracer: Tracer = Tracer("prepare_batch")

        if self._pipeline_config.sampling_config.enable_structured_output:
            assert self.vocab_size is not None
            bitmask = torch.full(
                xgr.get_bitmask_shape(
                    len(batch),
                    self.vocab_size,
                ),
                -1,
                dtype=torch.int32,
            )
        else:
            bitmask = None

        tracer.next("claim_cache_rows")
        for i, context in enumerate(batch):
            # Initialize a matcher if needed
            if context.json_schema and context.matcher is None:
                if not self._pipeline_config.sampling_config.enable_structured_output:
                    msg = "json_schema provided but constrained decoding is not enabled."
                    raise ValueError(msg)

                try:
                    compiled_grammar = (
                        self._grammar_compiler.compile_json_schema(
                            context.json_schema,
                            any_whitespace=False,
                        )
                    )
                    matcher = xgr.GrammarMatcher(compiled_grammar)
                    context.set_matcher(matcher)
                except Exception as e:
                    msg = f"Json schema provided in request cannot be compiled to valid grammar. \
                    Please update your json schema to produce valid structured output. From XGrammar: {e}"
                    logger.warning(msg)
                    # I am removing the json_schema, so it doesn't try to load the grammar repeatedly.
                    context.json_schema = None  # type: ignore

            if context.matcher:
                if (
                    jump_forward_string
                    := context.matcher.find_jump_forward_string()
                ):
                    tokens = self.tokenizer.encode(
                        jump_forward_string, add_special_tokens=False
                    )
                    for token in tokens:
                        context.jump_ahead(token)

            # Claim cache rows for context.
            if not self._pipeline_model.kv_manager.contains(
                context.cache_seq_id
            ):
                self._pipeline_model.kv_manager.external_claim(
                    [context.cache_seq_id]
                )

            # Update num_steps.
            num_steps = self.calculate_num_steps(num_steps, context)

            # Update bitmask
            if (
                self._pipeline_config.sampling_config.enable_structured_output
                and context.matcher
            ):
                context.matcher.fill_next_token_bitmask(bitmask, index=i)

        # `fetch` may shorten the input context by bumping the start_idx.
        tracer.next("fetch_kv_cache")
        kv_cache_inputs = self._pipeline_model.kv_manager.fetch(
            cast(list[InputContext], batch), num_steps
        )

        return (
            self._pipeline_model.prepare_initial_token_inputs(
                context_batch=batch,
                kv_cache_inputs=KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs
                ),
            ),
            num_steps,
            bitmask,
        )

    @traced
    def sample_logits(
        self,
        logits: Tensor,
        prev_tokens: Tensor,
        logit_offsets: Optional[Tensor],
        bitmask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        graph_inputs = [logits, prev_tokens]

        if logit_offsets:
            graph_inputs.append(logit_offsets)

        if bitmask:
            graph_inputs.append(bitmask)

        a, b = self._sampler(*graph_inputs)[:2]
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        return (a, b)

    @traced
    def next_token(
        self,
        batch: dict[str, T],
        num_steps: int,
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        tracer: Tracer = Tracer("compute_parameters")

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())

        # # Get extra compute parameters for each input.
        batch_top_n = [context.log_probabilities for context in context_batch]
        compute_log_probabilities = any(batch_top_n)
        batch_echo: list[bool] = [
            context.log_probabilities_echo for context in context_batch
        ]

        # Prepare the batch.
        model_inputs, num_steps, bitmask = self.prepare_batch(
            context_batch, num_steps
        )

        # Multistep execution loop.
        tracer.next("allocate_generated_tokens")
        generated_tokens = Tensor.zeros(
            (len(context_batch), 0),
            dtype=DType.int64,
            device=self._devices[0],
        )

        curr_step_inputs = model_inputs
        batch_log_probabilities = []
        tracer.next(f"multistep_execution_loop_{num_steps}_steps")
        for i in range(num_steps):
            tracer.push(f"step_{i}")

            # Execute the model and get next tokens.
            model_outputs = self._pipeline_model.execute(
                model_inputs=curr_step_inputs,
            )

            if bitmask is not None:
                assert self.vocab_size is not None
                bits = 2 ** torch.arange(32, dtype=torch.int32)
                bitmask = (bitmask.unsqueeze(-1) & bits) != 0
                bitmask = bitmask.reshape(
                    len(context_batch),
                    -1,
                ).to(torch.bool)
                bitmask = bitmask[:, 0 : self.vocab_size]

                bitmask = Tensor.from_dlpack(bitmask).to(self._devices[0])

            # Sample next token.
            tracer.next("sample_next_token")
            new_tokens, new_generated_tokens = self.sample_logits(
                model_outputs.logits,
                generated_tokens,
                model_outputs.logit_offsets,
                bitmask,
            )

            assert isinstance(new_tokens, Tensor)
            assert isinstance(new_generated_tokens, Tensor)
            generated_tokens = new_generated_tokens

            if compute_log_probabilities:
                try:
                    tracer.next("compute_log_probabilities")
                    batch_log_probabilities.append(
                        self._pipeline_model.compute_log_probabilities(
                            curr_step_inputs,
                            model_outputs,
                            new_tokens,
                            batch_top_n,
                            batch_echo,
                        )
                    )
                except NotImplementedError:
                    logger.warning(
                        "Unable to compute log probabilities for"
                        f" {self._pipeline_config.model_config.model_path}"
                    )
                    batch_log_probabilities.append(None)
            # Check if we're on our last iteration. If so, skip preparing the next batch
            if i == num_steps - 1:
                tracer.pop()  # pops f"step_{i}"
                break
            # Prepare inputs for the next token in multistep execution
            tracer.next("increment_cache_lengths")  # pops sample_next_token

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

            tracer.next("prepare_next_token_inputs")  # pops inc_cache_lengths
            curr_step_inputs = self._pipeline_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
            tracer.pop()  # pops step_{i}

        # Do the copy to host for each token generated.
        tracer.next(
            "generated_tokens.to(CPU())"
        )  # pops multistep_execution_loop_steps
        generated_tokens_host = generated_tokens.to_numpy()

        # Prepare the response, pruning away completed requests as we go.
        res: dict[str, TextGenerationResponse] = {}
        tracer.push("prepare_response")
        for batch_index, (request_id, context) in enumerate(batch.items()):
            status = TextGenerationStatus.ACTIVE
            res[request_id] = TextGenerationResponse([], status)
            for step in range(num_steps):
                # Convert to a Python scalar to improve serialization performance.
                next_token = int(generated_tokens_host[batch_index, step])

                # Get log probs if needed.
                log_probs: Optional[LogProbabilities] = None
                if compute_log_probabilities and (
                    log_probs_for_step := batch_log_probabilities[step]
                ):
                    log_probs = log_probs_for_step[batch_index]

                # Identify completion criteria.
                if context.ignore_eos:
                    is_eos = False
                else:
                    is_eos = next_token in self._eos_token_id

                # Write this token into our pre-allocated tokens array.
                context.update(
                    new_token=next_token,
                    log_probabilities=log_probs,
                    is_eos=is_eos,
                )

                max_length = upper_bounded_default(
                    upper_bound=self._pipeline_model.calculate_max_seq_len(
                        self._pipeline_config,
                        huggingface_config=self.huggingface_config,
                    ),
                    default=context.max_length,
                )

                if is_eos:
                    status = TextGenerationStatus.END_OF_SEQUENCE
                    res[request_id].update_status(status)
                elif context.current_length == max_length:
                    status = TextGenerationStatus.MAXIMUM_LENGTH
                    res[request_id].update_status(status)

                if status.is_done:
                    break

            # Walk outstanding completion tokens, and return to user.
            for token, log_probs in context.outstanding_completion_tokens():
                res[request_id].append_token(TextResponse(token, log_probs))

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        tracer.next("kv_manager.step")  # pops prepare_response
        self._pipeline_model.kv_manager.step(
            cast(list[InputContext], context_batch)
        )
        tracer.pop()  # pops kv_manager.step

        return res

    def release(self, context: T) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager."""
        self._pipeline_model.kv_manager.release(context.cache_seq_id)
