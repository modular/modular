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
"""MAX pipeline for model inference and generation (Text Generation variant).

This pipeline supports overlap scheduling where GPU execution is overlapped with
python host logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, load_devices
from max.graph import DeviceRef
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.nn.legacy.kv_cache import KVCacheInputsSequence
from max.nn.legacy.transformer import ReturnLogits
from max.profiler import Tracer, traced

from .utils import (
    calculate_num_steps,
    get_eos_tokens,
    get_weight_paths,
    update_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import PipelineConfig

from ..interfaces import PipelineModel
from ..interfaces.generate import GenerateMixin
from ..sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")


class OverlapTextGenerationPipeline(
    Pipeline[
        TextGenerationInputs[TextGenerationContextType], TextGenerationOutput
    ],
    GenerateMixin[TextGenerationContextType, TextGenerationRequest],
    Generic[TextGenerationContextType],
):
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

        model_config = pipeline_config.model
        huggingface_config = model_config.huggingface_config

        self._devices = load_devices(model_config.device_specs)
        self._tokenizer = tokenizer

        self._eos_token_id = get_eos_tokens(huggingface_config, eos_token_id)

        # Initialize Session.
        from max.engine import InferenceSession  # local import to avoid cycles

        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = get_weight_paths(model_config)

        # late imports to minimize header deps
        from max.graph.weights import load_weights as _load_weights
        from max.graph.weights import weights_format as _weights_format

        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            huggingface_config=huggingface_config,
            encoding=model_config.quantization_encoding,
            devices=self._devices,
            kv_cache_config=model_config.kv_cache,
            weights=_load_weights(weight_paths),
            adapter=weight_adapters.get(_weights_format(weight_paths)),
            return_logits=ReturnLogits.ALL
            if self._pipeline_config.enable_echo
            else ReturnLogits.LAST_TOKEN,
        )

        # Load sampler.
        self._sampler = session.load(
            token_sampler(
                self._pipeline_config.sampling,
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

    @property
    def kv_managers(
        self,
    ) -> list[Any]:
        """Return the list of KV cache managers backing this pipeline."""
        return [self._pipeline_model.kv_manager]

    @traced
    def prepare_batch(
        self,
        batches: list[list[TextGenerationContextType]],
        num_steps: int,
    ) -> tuple[
        Any,
        int,
        list[TextGenerationContextType],
    ]:
        """Prepare model inputs and ancillary state for multi-step execution.

        This flattens replica batches, ensures KV-cache reservations, clamps
        ``num_steps`` per context, and builds initial model inputs.

        Args:
            batches: Per-replica list of contexts.
            num_steps: Desired number of steps to run.

        Returns:
            A tuple of:
                - ModelInputs: Prepared inputs for the first step.
                - int: The clamped number of steps to run.
                - list[TextGenerationContextType]: The flattened context batch.
        """
        for replica_batch in batches:
            for context in replica_batch:
                # Update num_steps.
                num_steps = calculate_num_steps(
                    context, num_steps, self._pipeline_model.max_seq_len
                )

        # Retrieve the KV Cache Inputs.
        flat_batch = [context for batch in batches for context in batch]
        kv_cache_inputs = self._pipeline_model.kv_manager.get_runtime_inputs(
            batches, num_steps
        )

        replica_batches = batches
        return (
            self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=replica_batches,
                kv_cache_inputs=KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs
                ),
            ),
            num_steps,
            flat_batch,
        )

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """
        if inputs.enable_log_probs:
            raise ValueError(
                "Log probabilities are not supported with overlap pipeline"
            )

        if inputs.num_steps > 1:
            raise ValueError(
                "Max num steps > 1 is not supported with the Overlap scheduler."
            )

        device0 = self._devices[0]
        pinned = not device0.is_host
        # Prepare the batch.
        model_inputs, num_steps, flat_batch = self.prepare_batch(
            inputs.batches, inputs.num_steps
        )

        with Tracer("FusedSamplingProcessor"):
            sampling_processor = FusedSamplingProcessor(
                sampler=self._sampler,
                pipeline_config=self._pipeline_config,
                context_batch=flat_batch,
                num_steps=num_steps,
                device=device0,
            )

        with Tracer("pipeline_model.execute"):
            # Execute the model and get next tokens.
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=model_inputs
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
        assert model_outputs.logit_offsets is None

        # Sample next token unless this is an empty batch.
        # Empty batches still need to be run for deepseek DP / EP barrier.
        if len(flat_batch) > 0:
            with Tracer("sample_next_token"):
                apply_logits_processors(
                    context_batch=flat_batch,
                    batch_logits=model_outputs.logits,
                    batch_logit_offsets=model_outputs.logit_offsets,
                    batch_processors=[sampling_processor],
                )
                new_tokens = sampling_processor.new_tokens
                assert new_tokens is not None

        # Do the copy to host for each token generated.
        with Tracer("D2H generated_tokens"):
            generated_tokens_device = sampling_processor.generated_tokens
            # Allocate a pinned tensor on the host for faster async d2h transfer
            # speeds. If the model is on host, then fall back to normal pageable
            # memory.
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
        )

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        self._pipeline_model.kv_manager.step(flat_batch)

        return res

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager.

        Note: KV cache lifecycle is now managed by the scheduler. This method
        is kept for interface compatibility but is a no-op for regular pipelines.
        """
        # KV cache release is handled by the scheduler via batch_constructor
        pass
