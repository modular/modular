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
"""MAX pipeline for model inference and generation (Overlap Text Generation Variant).

This pipeline supports overlap scheduling where GPU execution is overlapped with
python host logic.

Note that this pipeline only supports num_steps=1.

Here is the CPU and GPU timeline for overlap scheduling:

   I3: Input processing for batch 3
   O3: Output processing for batch 3
   K3: GPU kernel execution for batch 3

    CPU: [I1][I2]          [O1][I3]      [O2][I4]      [O3][I5]      ...
    GPU:     [     K1     ][     K2     ][     K3     ][     K4     ][ ...

During I3, we have to prepare the model inputs for batch3. However, K2 may
still be in flight. If batch 2 and 3 share the same requests, then we rely on the
ScatterFutureTokenProcessor to prepare the ragged_input_tokens for batch 3 on
the GPU. This essentially scatters the generated tokens from the output of batch 2
on the slots corresponding to placeholder future tokens in batch 3's inputs.

For example:

  Batch 2 has reqA, reqB, reqC.
  Batch 3 has reqB, reqC, reqD.

    reqA = [I, dream, of, FUTURE_TOKEN]
    reqB = [I, like, go, FUTURE_TOKEN]
    reqC = [I, like, to, eat, FUTURE_TOKEN]
    reqD = [I, like, to, read]

  The ragged_input_tokens for Batch 3 would be:
                      idx=4                           idx=9
    [I, like, to, go, FUTURE_TOKEN, I, like, to, eat, FUTURE_TOKEN, I, like, to, read]

  ScatterFutureTokenProcessor would scatter the outputs of batch2 to the right slots:
    scatter_nd(
       inputs=ragged_input_tokens,
       indices=[[-99999], [4], [9]],
       updates=[sheep, fishing, cake]
    )

  Note that reqA is part of Batch 2 but not present in Batch 3. As such the update
  "sheep" corresponding to reqA is skipped since its idx=-99999 is out of bounds.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Dim, Graph, SymbolicDim, TensorType, ops
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    DUMMY_REQUEST_ID,
    BatchType,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.interfaces.tokens import TokenBuffer
from max.nn.legacy import kernels
from max.nn.legacy.kv_cache import KVCacheInputsSequence
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import Tracer, traced

from .text_generation import TextGenerationPipelineInterface
from .utils import (
    get_eos_tokens,
    get_weight_paths,
    update_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import PipelineConfig

from dataclasses import dataclass

from ..graph_capture import ServeGraphCaptureRunner
from ..interfaces import ModelInputs, ModelOutputs, PipelineModel
from ..sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")


@runtime_checkable
class _HasRaggedTokens(Protocol):
    tokens: Buffer


@runtime_checkable
class _SupportsModelCapture(Protocol):
    model: Model


@dataclass
class AsyncBatch(Generic[TextGenerationContextType]):
    """A batch that is being asynchronously executed on the GPU."""

    inputs: TextGenerationInputs[TextGenerationContextType]
    """The inputs for the batch."""

    generated_tokens_device: Buffer
    """The generated tokens for the batch on the gpu.

    The shape of the buffer is (batch_size,). The ordering of the generated tokens
    should be the same as the ordering of the requests in the input batch.
    """

    generated_tokens_host: Buffer
    """The generated tokens for the batch on the cpu.

    It is backed by pinned memory which makes d2h transfers asynchronous.
    This buffer is not ready to read until the batch has completed executing.

    This buffers has the same contents as `generated_tokens_device`.
    """

    _is_processed: bool = False
    """Whether the outputs have been already been processed."""

    @traced
    def sync_and_process_outputs(
        self,
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Syncs on completion of this batch and processes the outputs.

        Replaces the placeholder future tokens in the TextContext CPU numpy
        token buffers with the real token values.
        """
        if self._is_processed:
            raise ValueError("Outputs have already been processed.")
        self._is_processed = True

        # We assume that the call to `.to_numpy()` will insert a device
        # synchronize to guarantee that the async d2h transfer is done.
        generated_tokens_np = self.generated_tokens_host.to_numpy()

        # Now that we have synced, it is safe to read the contents of the
        # generated_tokens_np on the host.

        # Update the context object, realizing the placeholder future tokens.
        outputs = update_context_and_prepare_responses(
            generated_tokens_np,
            self.inputs.flat_batch,
            num_steps=1,
            overwrite_future=True,
        )

        return outputs


@traced
def build_scatter_future_tokens_graph() -> Graph:
    """Builds a trivial scatter_nd graph."""
    with Graph(
        "my_scatter_future_tokens_graph",
        input_types=[
            # Ragged input tokens for batch N
            TensorType(
                DType.int64,
                shape=[SymbolicDim("curr_batch_size")],
                device=DeviceRef.GPU(0),
            ),
            # Generated tokens for batch N-1
            TensorType(
                DType.int64,
                shape=[SymbolicDim("prev_batch_size"), Dim(1)],
                device=DeviceRef.GPU(0),
            ),
            # Indices that maps generated tokens from batch N-1 to batch N
            TensorType(
                DType.int32,
                shape=[SymbolicDim("prev_batch_size")],
                device=DeviceRef.GPU(0),
            ),
        ],
    ) as graph:
        tokens = graph.inputs[0].tensor
        future_token_inputs = graph.inputs[1].tensor
        future_token_indices = graph.inputs[2].tensor
        out = kernels.scatter_nd_skip_oob_indices(
            input=tokens,
            updates=ops.squeeze(future_token_inputs, axis=-1),
            indices=ops.unsqueeze(future_token_indices, axis=-1),
        )
        graph.output(out)
    return graph


class ScatterFutureTokenProcessor:
    """Processor for realizing placeholder future tokens in ragged input on the GPU.

    We scatter the generated tokens from the previous batch into the slots
    containing placeholder future tokens in the current batch. This all occurs
    efficiently on the gpu. We use a variant of scatter_nd that skips out of
    bound indices in cases where the current batch does not contain a request
    present in the previous batch.
    """

    def __init__(self, session: InferenceSession) -> None:
        self._scatter_future_tokens = session.load(
            build_scatter_future_tokens_graph()
        )

    @traced
    def scatter_future_tokens(
        self,
        prev_batch: AsyncBatch[TextGenerationContextType],
        inputs: TextGenerationInputs[TextGenerationContextType],
        ragged_input_tokens: Buffer,
    ) -> Buffer:
        """Scatters generated tokens from the previous batch into placeholder slots.

        Fills placeholder future tokens in the current batch on the GPU.
        """
        prev_generated_tokens = prev_batch.generated_tokens_device
        device = prev_generated_tokens.device
        if device.is_host:
            raise ValueError(
                "Scatter future tokens processor must be on the gpu."
            )
        if ragged_input_tokens.device != device:
            raise ValueError(
                "Ragged input tokens must be on the same device as the generated "
                "tokens from the previous batch."
            )

        # Prepare the scatter indices.
        prev_batch_size = prev_generated_tokens.shape[0]
        future_tok_indices = Buffer(
            shape=(prev_batch_size,),
            dtype=DType.int32,
            device=device,
            pinned=True,
        )
        future_tok_indices.disable_auto_sync()
        future_tok_indices_np = future_tok_indices.to_numpy()

        # Initialize the scatter indices with an oob_idx. These updates will be
        # skipped by the scatter_nd kernel.
        oob_idx = np.iinfo(np.int32).min
        future_tok_indices_np.fill(oob_idx)

        # If a request is present in both the previous and current batch,
        # then we must scatter the generated token into the placeholder future
        # token slot.
        prev_flat_batch = prev_batch.inputs.flat_batch
        req_id_to_idx_in_batch = {
            context.request_id: idx_in_batch
            for idx_in_batch, context in enumerate(inputs.flat_batch)
        }
        for prev_idx, context in enumerate(prev_flat_batch):
            req_id = context.request_id
            # If generated_length is still 0, then there is no placeholder future
            # token. This is possible due to chunked prefill.
            if (
                req_id in req_id_to_idx_in_batch
                and context.tokens.generated_length
            ):
                future_tok_indices_np[prev_idx] = req_id_to_idx_in_batch[req_id]

        # Execute the scatter_nd kernel.
        (new_ragged_input_tokens,) = self._scatter_future_tokens.execute(
            ragged_input_tokens,
            prev_generated_tokens,
            future_tok_indices.to(device),
        )

        return new_ragged_input_tokens


class OverlapTextGenerationPipeline(
    TextGenerationPipelineInterface[TextGenerationContextType],
    Generic[TextGenerationContextType],
):
    """Overlap text generation pipeline."""

    _pipeline_model: PipelineModel[Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
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
        if huggingface_config is None:
            raise ValueError(
                f"Overlap text generation pipeline requires a HuggingFace config for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        self._devices = load_devices(model_config.device_specs)
        if self._devices[0].is_host:
            raise ValueError(
                "OverlapTextGenerationPipeline does not support CPU models."
            )
        self._tokenizer = tokenizer

        self._eos_token_id = get_eos_tokens(huggingface_config, eos_token_id)

        session = InferenceSession(devices=self._devices)
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        if not model_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = get_weight_paths(model_config)

        self._pipeline_model: PipelineModel[Any] = pipeline_model(
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

        # Load sampler.
        self._sampler = session.load(
            token_sampler(
                self._pipeline_config.sampling,
                device=DeviceRef.from_device(self._devices[0]),
            )
        )

        self._kv_manager = self._pipeline_model.kv_manager

        # Overlap scheduling specific initialization.

        # Load the scatter future tokens graph.
        self._scatter_future_tokens = ScatterFutureTokenProcessor(session)
        # Set previous asynchronously executing batch to None.
        self._prev_batch: AsyncBatch[TextGenerationContextType] | None = None
        self._graph_capture_runner: ServeGraphCaptureRunner | None = None

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
        return [self._kv_manager]

    def has_pending_outputs(self) -> bool:
        """Returns True if there are pending outputs for the previous batch.

        If this is True, the caller should call `.execute()` even with empty
        inputs to retrieve the outputs for the previous batch.
        """
        return self._prev_batch is not None

    @contextmanager
    def _warmup_model_inputs(self, batch_size: int) -> Iterator[ModelInputs]:
        capture_contexts = [
            TextContext(
                max_length=self._pipeline_model.max_seq_len,
                tokens=TokenBuffer(np.zeros(1, dtype=np.int64)),
                eos_token_ids=self._eos_token_id,
                model_name=self._pipeline_config.model.model_name,
                request_id=RequestID(
                    f"{DUMMY_REQUEST_ID}-overlap-capture-b{batch_size}-i{idx}"
                ),
            )
            for idx in range(batch_size)
        ]
        with self._kv_manager.reserve(
            capture_contexts, replica_idx=0, num_steps=1
        ):
            kv_cache_inputs = self._kv_manager.get_runtime_inputs(
                [capture_contexts], num_steps=1
            )
            with Tracer("prepare_initial_token_inputs"):
                model_inputs = (
                    self._pipeline_model.prepare_initial_token_inputs(
                        replica_batches=[capture_contexts],
                        kv_cache_inputs=KVCacheInputsSequence(
                            kv_cache_inputs=kv_cache_inputs
                        ),
                    )
                )
            yield model_inputs

    def warmup_graph_capture(self) -> None:
        """Initializes and runs overlap device graph capture warmup."""
        if not isinstance(self._pipeline_model, _SupportsModelCapture):
            raise RuntimeError(
                "Device graph capture is enabled but pipeline model does not "
                "expose a compiled model for capture/replay."
            )
        if self._pipeline_config.max_batch_size is None:
            raise RuntimeError(
                "device_graph_capture requires max_batch_size to be resolved."
            )

        graph_capture_runner = ServeGraphCaptureRunner(
            model=self._pipeline_model.model,
            warmup_model_inputs=self._warmup_model_inputs,
            execute_model=self._pipeline_model.execute,
            max_batch_size=self._pipeline_config.max_batch_size,
        )
        self._graph_capture_runner = graph_capture_runner
        logger.info("Starting serve device graph capture warmup.")
        graph_capture_runner.warmup_pre_ready()
        logger.info("Completed serve device graph capture warmup.")

    def _run_forward(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> ModelOutputs:
        """Runs the forward pass for the provided inputs and returns the ModelOutputs."""
        # Prepare the batch.
        kv_cache_inputs = self._kv_manager.get_runtime_inputs(
            inputs.batches, num_steps=1
        )

        with Tracer("prepare_initial_token_inputs"):
            model_inputs = self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=inputs.batches,
                kv_cache_inputs=KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs
                ),
            )

        if not isinstance(model_inputs, _HasRaggedTokens):
            raise RuntimeError(
                "OverlapTextGenerationPipeline requires model inputs with a "
                "Buffer `tokens` field."
            )
        ragged_input_tokens = model_inputs.tokens
        if self._prev_batch is not None:
            with Tracer("scatter_future_tokens"):
                new_ragged_input_tokens = (
                    self._scatter_future_tokens.scatter_future_tokens(
                        prev_batch=self._prev_batch,
                        inputs=inputs,
                        ragged_input_tokens=ragged_input_tokens,
                    )
                )
            # Overwrite the ragged input tokens with the new ones.
            model_inputs.tokens = new_ragged_input_tokens

        # Execute the model and get next tokens.
        try:
            with Tracer("pipeline_model.execute"):
                if (
                    (runner := self._graph_capture_runner)
                    and inputs
                    and inputs.batch_type == BatchType.TG
                ):
                    return runner.replay(model_inputs=model_inputs)

                return self._pipeline_model.execute(model_inputs=model_inputs)
        except Exception:
            batch_size = len(inputs.flat_batch)
            cache_tokens = sum(
                ctx.tokens.processed_length for ctx in inputs.flat_batch
            )
            input_tokens = sum(
                ctx.tokens.active_length for ctx in inputs.flat_batch
            )
            logger.error(
                "Encountered an exception while executing batch: "
                f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}"
            )
            raise  # re-raise the original exception

    def _run_forward_and_sample_logits(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> AsyncBatch[TextGenerationContextType]:
        """Runs the forward pass for the provided inputs and returns the AsyncBatch."""
        device0 = self._devices[0]
        assert not device0.is_host

        flat_batch = inputs.flat_batch
        with Tracer("FusedSamplingProcessor"):
            sampling_processor = FusedSamplingProcessor(
                sampler=self._sampler,
                pipeline_config=self._pipeline_config,
                context_batch=flat_batch,
                num_steps=1,
                device=device0,
            )

        model_outputs = self._run_forward(inputs)

        curr_batch: AsyncBatch[TextGenerationContextType] | None = None
        with Tracer("apply_logits_processors"):
            apply_logits_processors(
                context_batch=flat_batch,
                batch_logits=model_outputs.logits,
                batch_logit_offsets=model_outputs.logit_offsets,
                batch_processors=[sampling_processor],
            )
        generated_tokens_device = sampling_processor.generated_tokens

        # Do the copy to host for each token generated.
        with Tracer("D2H generated_tokens"):
            # Allocate a pinned tensor on the host for faster async d2h transfer
            # speeds.
            generated_tokens_host = Buffer(
                shape=generated_tokens_device.shape,
                dtype=generated_tokens_device.dtype,
                device=device0,
                pinned=True,
            )
            generated_tokens_host.disable_auto_sync()
            generated_tokens_host.inplace_copy_from(generated_tokens_device)
            # Record an event associated with the buffer to track the
            # completion of the d2h copy.
            # This will ensure that the subsequent call to `to_numpy()` will
            # block until the d2h copy is complete, and no more.
            generated_tokens_host.mark_as_ready()

        curr_batch = AsyncBatch(
            inputs=inputs,
            generated_tokens_device=generated_tokens_device,
            generated_tokens_host=generated_tokens_host,
        )
        return curr_batch

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Executes a batch of requests asynchronously on the GPU.

        This method will return before the outputs for the current batch are ready.
        The caller may need to call `.execute()` again (possibly with an empty batch)
        to retrieve these outputs. For example:

        ```python
        output_a = pipeline.execute(inputs)
        assert len(outputs) == 0

        output_b = pipeline.execute(empty_inputs)
        assert len(outputs) == len(inputs.flat_batch)
        ```

        Args:
            inputs: The inputs for the batch.

        Returns:
            A dictionary of request IDs to outputs. The outputs will not correspond
            to the requests in the input batch. Instead they are of the previous batch.
        """
        if inputs.enable_log_probs:
            raise ValueError(
                "Log probabilities are not supported with overlap pipeline"
            )

        if inputs.num_steps > 1:
            raise ValueError(
                "Max num steps > 1 is not supported with the Overlap scheduler."
            )

        if inputs:
            # Run the entire forward pass and output processing if the batch has
            # at least one request.
            curr_batch = self._run_forward_and_sample_logits(inputs)
        elif self.pipeline_config.execute_empty_batches:
            # If the batch is empty and execute_empty_batches is True, we will
            # only run the forward pass to ensure that the barrier point is reached
            # for EP + DP. We skip all output processing.
            _ = self._run_forward(inputs)
            curr_batch = None
        else:
            curr_batch = None

        if self._prev_batch is not None:
            outputs: PipelineOutputsDict[TextGenerationOutput] = (
                self._prev_batch.sync_and_process_outputs()
            )

            self._prev_batch = None
        else:
            # Empty outputs as there is no previous batch.
            outputs = {}

        for context in inputs.flat_batch:
            context.update_with_future_token()

        # Commit the new KV blocks into the prefix cache, ignoring the final
        # placeholder future token.
        self._kv_manager.step(inputs.batches)

        if curr_batch is not None:
            self._prev_batch = curr_batch

        return outputs

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager.

        Note: KV cache lifecycle is now managed by the scheduler. This method
        is kept for interface compatibility but is a no-op for regular pipelines.
        """
        # KV cache release is handled by the scheduler via batch_constructor
        pass
