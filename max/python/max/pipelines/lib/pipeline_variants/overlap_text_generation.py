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
RealizeFutureTokenProcessor to prepare the ragged_input_tokens for batch 3 on
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

  RealizeFutureTokenProcessor would scatter the outputs of batch2 to the right slots:
    scatter_nd(
       inputs=ragged_input_tokens,
       indices=[[-99999], [4], [9]],
       updates=[sheep, fishing, cake]
    )

  Note that reqA is part of Batch 2 but not present in Batch 3. As such the update
  "sheep" corresponding to reqA is skipped since its idx=-99999 is out of bounds.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    final,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, DeviceEvent, DevicePinnedBuffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    SymbolicDim,
    TensorType,
    TensorValue,
    ops,
)
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    BatchType,
    EOSTracker,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    SpecDecodingState,
    TextGenerationContextType,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.interfaces.tokens import TokenBuffer
from max.kv_cache import PagedKVCacheManager, load_multi_kv_managers
from max.kv_cache.paged_kv_cache.cache_manager import _contiguous_prefix_2d
from max.nn import kernels
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    shape_to_scalar,
)
from max.profiler import Tracer, traced

from ..speculative_decoding.base import SpeculativeDecodingMetrics
from .text_generation import TextGenerationPipelineInterface, load_kv_manager
from .utils import (
    get_eos_tokens,
    update_context_and_prepare_responses,
    update_spec_decode_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

from dataclasses import dataclass

from ..graph_capture import ServeGraphCaptureRunner
from ..interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
)
from ..sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

logger = logging.getLogger("max.pipelines")

_MAX_GRAPH_CAPTURE_BATCH_SIZE = 128
_OOB_IDX = np.iinfo(np.int32).min
_MAGIC_DRAFT_TOKEN_ID = 42


@runtime_checkable
class _UnifiedEagleInputs(Protocol):
    tokens: Buffer
    input_row_offsets: Buffer
    kv_cache_inputs: KVCacheInputs[Buffer, Buffer]

    draft_tokens: Buffer | None
    draft_kv_blocks: list[Buffer] | None


def _get_draft_kv_blocks(
    draft_kv_manager: PagedKVCacheManager,
    data_parallel_degree: int,
) -> list[Buffer]:
    """Extract persistent draft KV block buffers (one per device).

    cache_lengths are NOT saved here — they must be created fresh
    per-execute to match the runtime batch size.
    """
    draft_kv_inputs = draft_kv_manager.runtime_inputs(
        [[] for _ in range(data_parallel_degree)]
    )
    return [per_dev.kv_blocks for per_dev in draft_kv_inputs.inputs]


@dataclass
class SpecDecodeState:
    """Pipeline for unified EAGLE: single fused graph handles target + draft.

    Unlike EAGLESpeculativeDecodingPipeline which manages two separate models,
    this pipeline uses a single model that runs both target forward and draft
    generation in one compiled graph call. Rejection sampling also happens
    in-graph (greedy acceptance).

    Orchestration:
    Prefill: model(draft_tokens=[?,0]) -> commit bonus, save new_token.
    Decode:  model(draft_tokens=[?,K]) -> verify drafts, commit tokens,
            save new_token for next iteration.
    """

    num_speculative_tokens: int
    """The number of speculative tokens to generate."""

    target_kv_manager: PagedKVCacheManager
    """The KVCache manager for the target model."""

    draft_kv_blocks: list[Buffer]
    """The KVCache blocks for the draft model."""

    metrics: SpeculativeDecodingMetrics
    """The metrics for speculative decoding."""

    persistent_draft_tokens: Buffer
    """Persistent input buffer for draft tokens.

    A stable buffer must be used for inputs to device graphs."""

    @classmethod
    def load(
        cls,
        session: InferenceSession,
        model: PipelineModelWithKVCache[Any],
        pipeline_config: PipelineConfig,
    ) -> SpecDecodeState:
        """Load the spec decode state."""
        if pipeline_config.speculative is None:
            raise ValueError(
                "Speculative decoding is not enabled in the pipeline config."
            )

        target_kv_params = model.kv_params
        assert isinstance(target_kv_params, KVCacheParams)
        assert hasattr(model, "_draft_kv_params"), "Draft KV params not found"
        draft_kv_params = model._draft_kv_params
        assert isinstance(draft_kv_params, KVCacheParams)

        multi_kv_params = MultiKVCacheParams.from_params(
            target_kv_params, draft_kv_params
        )
        target_kv_mgr, draft_kv_mgr = load_multi_kv_managers(
            params=multi_kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=model.max_seq_len,
            session=session,
            available_cache_memory=pipeline_config.model.kv_cache._available_cache_memory,
        )
        target_kv_manager = target_kv_mgr

        draft_kv_blocks = _get_draft_kv_blocks(
            draft_kv_mgr, multi_kv_params.data_parallel_degree
        )
        assert len(draft_kv_blocks) == target_kv_params.n_devices

        num_speculative_tokens = (
            pipeline_config.speculative.num_speculative_tokens
        )
        spec_decoding_metrics = SpeculativeDecodingMetrics.empty(
            num_speculative_tokens=num_speculative_tokens
        )

        assert pipeline_config.runtime.max_batch_size is not None
        persistent_draft_tokens = Buffer(
            dtype=DType.int64,
            shape=(
                pipeline_config.runtime.max_batch_size
                * pipeline_config.model.data_parallel_degree,
                num_speculative_tokens,
            ),
            device=model.devices[0],
        )

        return SpecDecodeState(
            num_speculative_tokens=num_speculative_tokens,
            target_kv_manager=target_kv_manager,
            draft_kv_blocks=draft_kv_blocks,
            metrics=spec_decoding_metrics,
            persistent_draft_tokens=persistent_draft_tokens,
        )


@runtime_checkable
class _HasRaggedTokens(Protocol):
    tokens: Buffer
    input_row_offsets: Buffer


@runtime_checkable
class _SupportsModelCapture(Protocol):
    model: Model


@dataclass
class _AsyncBatchOutput:
    output_dict: PipelineOutputsDict[TextGenerationOutput]
    spec_decode_metrics: SpeculativeDecodingMetrics | None = None


@dataclass
class AsyncSpecDecodeBatch:
    """Extra outputs specific for speculative decoding async batch."""

    draft_tokens_to_verify_device: Buffer
    """The draft tokens to verify for the batch on gpu.

    The shape of the buffer is (batch_size, num_draft_tokens_to_verify).
    """

    draft_tokens_to_verify_host: Buffer
    """The draft tokens to verify for the batch.

    The shape of the array is (batch_size, num_draft_tokens_to_verify).
    """

    next_draft_tokens_device: Buffer
    """The next draft tokens for the batch on gpu.

    The shape of the buffer is (batch_size, num_speculative_tokens).
    """

    next_draft_tokens_host: Buffer
    """The next draft tokens for the batch on pinned gpu memory.

    The shape of the buffer is (batch_size, num_speculative_tokens).
    """

    num_accepted_draft_tokens_device: Buffer
    """The number of accepted draft tokens for the batch on gpu.

    The shape of the buffer is (batch_size,).
    """

    num_accepted_draft_tokens_host: Buffer
    """The number of accepted draft tokens for the batch on pinned gpu memory.

    The shape of the buffer is (batch_size,).
    """

    max_seq_len: int
    """The maximum sequence length for the pipeline model."""

    @property
    def num_draft_tokens_to_verify(self) -> int:
        """The number of draft tokens to verify during this batch."""
        return self.draft_tokens_to_verify_host.shape[1]


@dataclass
class AsyncBatch(Generic[TextGenerationContextType]):
    """A batch that is being asynchronously executed on the GPU."""

    inputs: TextGenerationInputs[TextGenerationContextType]
    """The inputs for the batch.
    """

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

    copy_event: DeviceEvent
    """Event that tracks completion of the d2h copy."""

    _is_processed: bool = False
    """Whether the outputs have been already been processed."""

    spec_decode: AsyncSpecDecodeBatch | None = None
    """Extra outputs specific for speculative decoding async batch."""

    @traced
    def sync_and_process_outputs(
        self,
    ) -> _AsyncBatchOutput:
        """Syncs on completion of this batch and processes the outputs.

        Replaces the placeholder future tokens in the TextContext CPU numpy
        token buffers with the real token values.
        """
        if self._is_processed:
            raise ValueError("Outputs have already been processed.")
        self._is_processed = True

        # Synchronize on the copy event to ensure the async d2h transfer is done.
        self.copy_event.synchronize()
        generated_tokens_np = self.generated_tokens_host.to_numpy()

        # Now that we have synced, it is safe to read the contents of the
        # generated_tokens_np on the host.

        if self.spec_decode is None:
            # Update the context object, realizing the placeholder future tokens.
            batch_size = len(self.inputs.flat_batch)
            assert generated_tokens_np.shape == (batch_size,)
            outputs = update_context_and_prepare_responses(
                generated_tokens_np.reshape((batch_size, 1)),
                self.inputs.flat_batch,
                num_steps=1,
                overwrite_future=True,
            )
            wrapped_outputs = _AsyncBatchOutput(output_dict=outputs)
        else:
            spec_decode_batch = self.spec_decode
            draft_tokens_np = (
                spec_decode_batch.draft_tokens_to_verify_host.to_numpy()
            )
            num_accepted_draft_tokens = (
                spec_decode_batch.num_accepted_draft_tokens_host.to_numpy()
            )
            next_draft_tokens = (
                self.spec_decode.next_draft_tokens_host.to_numpy()
            )
            max_seq_len = spec_decode_batch.max_seq_len

            outputs = update_spec_decode_context_and_prepare_responses(
                draft_tokens=draft_tokens_np,
                next_draft_tokens=next_draft_tokens,
                num_accepted_draft_tokens=num_accepted_draft_tokens,
                next_tokens=generated_tokens_np,
                context_batch=self.inputs.flat_batch,
                max_seq_len=max_seq_len,
            )

            batch_size = len(self.inputs.flat_batch)
            is_dummy_draft_tokens: list[bool] = [
                all(draft_tokens_np[batch_idx, :] == _MAGIC_DRAFT_TOKEN_ID)
                for batch_idx in range(batch_size)
            ]

            num_speculative_tokens = next_draft_tokens.shape[1]
            num_draft_tokens_to_verify = draft_tokens_np.shape[1]

            # Compute per-position acceptance counts.
            # For each position i, count how many requests accepted at least i+1 tokens.
            accepted_per_position = [0] * num_draft_tokens_to_verify
            for is_dummy, accepted_count in zip(
                is_dummy_draft_tokens, num_accepted_draft_tokens, strict=True
            ):
                if is_dummy:
                    continue
                for pos in range(int(accepted_count)):
                    if pos < num_draft_tokens_to_verify:
                        accepted_per_position[pos] += 1

            # Only count verifications when there are real draft tokens to verify.
            # Otherwise we'd dilute the per-position acceptance rate.
            num_verifications = (
                batch_size - sum(is_dummy_draft_tokens)
                if num_draft_tokens_to_verify > 0
                else 0
            )

            metrics = SpeculativeDecodingMetrics(
                num_speculative_tokens=num_speculative_tokens,
                accepted_per_position=accepted_per_position,
                num_verifications=num_verifications,
            )

            wrapped_outputs = _AsyncBatchOutput(
                output_dict=outputs,
                spec_decode_metrics=metrics,
            )

        return wrapped_outputs


_Tensor = TypeVar("_Tensor")
_Buffer = TypeVar("_Buffer")


@dataclass
class _RealizeFutureTokenSpecDecodeInputs(Generic[_Tensor, _Buffer]):
    curr_draft_tokens: _Tensor
    data_parallel_splits: _Tensor | None
    curr_cache_lengths: Sequence[_Tensor]
    signal_buffers: Sequence[_Buffer] | None
    prev_generated_draft_tokens: _Tensor
    prev_draft_tokens: _Tensor
    prev_num_accepted_draft_tokens: _Tensor

    def flatten(self) -> list[_Tensor | _Buffer]:
        return [
            self.curr_draft_tokens,
            *(
                (self.data_parallel_splits,)
                if self.data_parallel_splits is not None
                else ()
            ),
            *self.curr_cache_lengths,
            *(self.signal_buffers if self.signal_buffers is not None else ()),
            self.prev_generated_draft_tokens,
            self.prev_draft_tokens,
            self.prev_num_accepted_draft_tokens,
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> _RealizeFutureTokenSpecDecodeInputs[Any, Any]:
        return _RealizeFutureTokenSpecDecodeInputs(
            curr_draft_tokens=next(it),
            data_parallel_splits=next(it)
            if self.data_parallel_splits is not None
            else None,
            curr_cache_lengths=[
                next(it) for _ in range(len(self.curr_cache_lengths))
            ],
            signal_buffers=[next(it) for _ in range(len(self.signal_buffers))]
            if self.signal_buffers is not None
            else None,
            prev_generated_draft_tokens=next(it),
            prev_draft_tokens=next(it),
            prev_num_accepted_draft_tokens=next(it),
        )


@dataclass
class _RealizeFutureTokenInputs(Generic[_Tensor, _Buffer]):
    prev_to_curr_map: _Tensor
    curr_to_prev_map: _Tensor
    curr_tokens: _Tensor
    curr_input_row_offsets: _Tensor
    prev_generated_tokens: _Tensor

    spec_decode: (
        _RealizeFutureTokenSpecDecodeInputs[_Tensor, _Buffer] | None
    ) = None

    def flatten(self) -> list[_Tensor | _Buffer]:
        return [
            self.prev_to_curr_map,
            self.curr_to_prev_map,
            self.curr_tokens,
            self.curr_input_row_offsets,
            self.prev_generated_tokens,
            *(
                self.spec_decode.flatten()
                if self.spec_decode is not None
                else ()
            ),
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> _RealizeFutureTokenInputs[Any, Any]:
        return _RealizeFutureTokenInputs(
            prev_to_curr_map=next(it),
            curr_to_prev_map=next(it),
            curr_tokens=next(it),
            curr_input_row_offsets=next(it),
            prev_generated_tokens=next(it),
            spec_decode=self.spec_decode.unflatten(it)
            if self.spec_decode is not None
            else None,
        )


def build_realize_future_token_graph(
    *,
    devices: Sequence[DeviceRef],
    enable_dp: int,
    num_speculative_tokens: int,
) -> Graph:
    """Builds a graph that prepares the input for the next batch."""
    device0 = devices[0]
    if num_speculative_tokens > 0:
        spec_decode_input_types = _RealizeFutureTokenSpecDecodeInputs[
            TensorType, BufferType
        ](
            curr_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("curr_batch_size"),
                    SymbolicDim("num_draft_tokens"),
                ],
                device=device0,
            ),
            data_parallel_splits=TensorType(
                DType.int64,
                shape=[SymbolicDim("num_replicas_plus_one")],
                device=DeviceRef.CPU(),
            )
            if enable_dp
            else None,
            curr_cache_lengths=[
                TensorType(
                    DType.uint32,
                    shape=[SymbolicDim(f"curr_batch_size_gpu_{i}")],
                    device=device,
                )
                for i, device in enumerate(devices)
            ],
            signal_buffers=[
                BufferType(
                    DType.uint8,
                    shape=[SymbolicDim(f"signal_buffer_gpu_{i}")],
                    device=device,
                )
                for i, device in enumerate(devices)
            ]
            if len(devices) > 1
            else None,
            prev_generated_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("prev_batch_size"),
                    SymbolicDim("num_draft_tokens"),
                ],
                device=device0,
            ),
            prev_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("prev_batch_size"),
                    SymbolicDim("prev_num_draft_tokens"),
                ],
                device=device0,
            ),
            prev_num_accepted_draft_tokens=TensorType(
                DType.int64,
                shape=[SymbolicDim("prev_batch_size")],
                device=device0,
            ),
        )
    else:
        spec_decode_input_types = None
    input_types = _RealizeFutureTokenInputs[TensorType, BufferType](
        prev_to_curr_map=TensorType(
            DType.int64,
            shape=[SymbolicDim("prev_batch_size")],
            device=device0,
        ),
        curr_to_prev_map=TensorType(
            DType.int64,
            shape=[SymbolicDim("curr_batch_size")],
            device=device0,
        ),
        curr_tokens=TensorType(
            DType.int64,
            shape=[SymbolicDim("seq_len")],
            device=device0,
        ),
        curr_input_row_offsets=TensorType(
            DType.uint32,
            shape=[SymbolicDim("curr_batch_size_plus_one")],
            device=device0,
        ),
        prev_generated_tokens=TensorType(
            DType.int64,
            shape=[SymbolicDim("prev_batch_size")],
            device=device0,
        ),
        spec_decode=spec_decode_input_types,
    )
    with Graph(
        "realize_future_token_graph",
        input_types=input_types.flatten(),
    ) as graph:
        it = iter(graph.inputs)
        input_values = input_types.unflatten(it)

        curr_to_prev_map = ops.unsqueeze(input_values.curr_to_prev_map, axis=-1)
        prev_to_curr_map = ops.unsqueeze(input_values.prev_to_curr_map, axis=-1)

        curr_input_row_offsets = ops.rebind(
            input_values.curr_input_row_offsets, [Dim("curr_batch_size") + 1]
        )
        possible_future_token_indices = ops.rebind(
            curr_input_row_offsets[1:] - 1, ["curr_batch_size"]
        )

        oob_idx = ops.constant(_OOB_IDX, dtype=DType.int64, device=device0)

        # scatter the prev generated tokens into the curr tokens
        prev_to_curr_token_indices = oob_idx.broadcast_to(("prev_batch_size",))
        prev_to_curr_token_indices = kernels.scatter_nd_skip_oob_indices(
            input=prev_to_curr_token_indices,
            updates=possible_future_token_indices.cast(DType.int64),
            indices=curr_to_prev_map,
        )
        realized_tokens = kernels.scatter_nd_skip_oob_indices(
            input=input_values.curr_tokens,
            updates=input_values.prev_generated_tokens,
            indices=ops.unsqueeze(prev_to_curr_token_indices, axis=-1),
        )

        if input_values.spec_decode is None:
            graph.output(realized_tokens)
        else:
            spec_decode = input_values.spec_decode
            num_draft_tokens_dim = (
                spec_decode.prev_generated_draft_tokens.shape[1]
            )
            num_draft_tokens = shape_to_scalar(
                num_draft_tokens_dim, device0, dtype=DType.uint32
            )

            # 0...K
            draft_col_range = ops.range(
                start=0,
                stop=num_draft_tokens_dim,
                out_dim="num_draft_tokens",
                device=device0,
                dtype=DType.uint32,
            )

            draft_slot_indices = (
                prev_to_curr_map * num_draft_tokens
            ).broadcast_to(
                shape=["prev_batch_size", "num_draft_tokens"]
            ) + draft_col_range

            total_curr_draft_elems = SymbolicDim(
                "curr_batch_size"
            ) * SymbolicDim("num_draft_tokens")
            total_prev_draft_elems = SymbolicDim(
                "prev_batch_size"
            ) * SymbolicDim("num_draft_tokens")
            realized_draft_tokens = kernels.scatter_nd_skip_oob_indices(
                input=spec_decode.curr_draft_tokens.reshape(
                    [total_curr_draft_elems]
                ),
                updates=spec_decode.prev_generated_draft_tokens.reshape(
                    [total_prev_draft_elems]
                ),
                indices=draft_slot_indices.reshape([total_prev_draft_elems, 1]),
            ).reshape(["curr_batch_size", "num_draft_tokens"])

            # Per-sequence increments in int64 (may be negative), then fold into
            # uint32 cache lengths to match :class:`~max.nn.kv_cache.KVCacheParams`
            # paged-cache ``cache_lengths`` dtype.
            batch_increments_i64 = ops.broadcast_to(
                ops.constant(0, dtype=DType.int64, device=device0),
                ["curr_batch_size"],
            )

            # The curr cache lengths already account for the draft tokens optimistically
            # (full speculative depth). Subtract that depth using the configured
            # ``num_speculative_tokens``, not the realize graph's draft-column count:
            # when the current batch passes ``draft_tokens`` with shape ``(B, 0)`` we
            # still must subtract the full K used for optimistic cache extension.
            prev_num_draft_tokens = shape_to_scalar(
                spec_decode.prev_draft_tokens.shape[1],
                device0,
                dtype=DType.int64,
            )
            delta = (
                spec_decode.prev_num_accepted_draft_tokens
                - ops.broadcast_to(prev_num_draft_tokens, ["prev_batch_size"])
            )
            batch_increments_i64 = kernels.scatter_nd_skip_oob_indices(
                input=batch_increments_i64,
                updates=delta,
                indices=prev_to_curr_map,
            )

            curr_cache_lenghts = spec_decode.curr_cache_lengths

            # realize the cache lengths
            realized_cache_lengths: list[TensorValue] = []
            if len(devices) == 1:
                cache_length_u32 = ops.rebind(
                    curr_cache_lenghts[0], ["curr_batch_size"]
                )
                cache_length_adjusted = (
                    cache_length_u32.cast(DType.int64) + batch_increments_i64
                )
                realized_cache_lengths.append(
                    cache_length_adjusted.cast(DType.uint32)
                )
            elif enable_dp:
                # DP > 1
                assert spec_decode.signal_buffers is not None
                assert spec_decode.data_parallel_splits is not None

                batch_increments_distributed = ops.distributed_broadcast(
                    batch_increments_i64, spec_decode.signal_buffers
                )

                for i in range(len(devices)):
                    start_offset = spec_decode.data_parallel_splits[i]
                    end_offset = spec_decode.data_parallel_splits[i + 1]

                    batch_increments_local = ops.slice_tensor(
                        batch_increments_distributed[i],
                        [
                            (
                                slice(
                                    start_offset,
                                    end_offset,
                                ),
                                f"curr_batch_size_gpu_{i}",
                            )
                        ],
                    )
                    replica_cache_length = (
                        curr_cache_lenghts[i].cast(DType.int64)
                        + batch_increments_local
                    )
                    realized_cache_lengths.append(
                        replica_cache_length.cast(DType.uint32)
                    )
            else:
                # TP > 1
                assert spec_decode.signal_buffers is not None
                cache_length_u32 = ops.rebind(
                    curr_cache_lenghts[0], ["curr_batch_size"]
                )
                cache_length_adjusted = (
                    cache_length_u32.cast(DType.int64) + batch_increments_i64
                ).cast(DType.uint32)

                realized_cache_lengths.extend(
                    ops.distributed_broadcast(
                        cache_length_adjusted, spec_decode.signal_buffers
                    )
                )

            graph.output(
                realized_tokens,
                realized_draft_tokens,
                *realized_cache_lengths,
            )

    return graph


class RealizeFutureTokenProcessor:
    """Processor for realizing placeholder future tokens in ragged input on the GPU.

    We scatter the generated tokens from the previous batch into the slots
    containing placeholder future tokens in the current batch. This all occurs
    efficiently on the gpu. We use a variant of scatter_nd that skips out of
    bound indices in cases where the current batch does not contain a request
    present in the previous batch.
    """

    def __init__(
        self,
        session: InferenceSession,
        devices: Sequence[DeviceRef],
        num_speculative_tokens: int = 0,
        enable_dp: bool = False,
    ) -> None:
        self._graph = session.load(
            build_realize_future_token_graph(
                devices=devices,
                num_speculative_tokens=num_speculative_tokens,
                enable_dp=enable_dp,
            )
        )
        self._enable_dp = enable_dp
        self._num_speculative_tokens = num_speculative_tokens

    def _compute_mappings(
        self,
        prev_batch: AsyncBatch[TextGenerationContextType],
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> tuple[DevicePinnedBuffer, DevicePinnedBuffer] | None:
        """Computes scatter indices mapping previous-batch tokens to current slots.

        Returns None if all indices are out-of-bounds (no overlap between
        the previous and current batch), indicating the scatter can be skipped.
        """
        prev_generated_tokens = prev_batch.generated_tokens_device
        device = prev_generated_tokens.device
        if device.is_host:
            raise ValueError(
                "Realize future tokens processor must be on the gpu."
            )

        # Prepare the scatter indices.
        prev_batch_size = prev_generated_tokens.shape[0]
        prev_to_curr_map_host = DevicePinnedBuffer(
            shape=(prev_batch_size,),
            dtype=DType.int64,
            device=device,
        )
        prev_to_curr_map = prev_to_curr_map_host.to_numpy()

        curr_batch_size = len(inputs.flat_batch)
        curr_to_prev_map_host = DevicePinnedBuffer(
            shape=(curr_batch_size,),
            dtype=DType.int64,
            device=device,
        )
        curr_to_prev_map = curr_to_prev_map_host.to_numpy()

        # Initialize the scatter indices with an oob_idx. These updates will be
        # skipped by the scatter_nd kernel.
        prev_to_curr_map.fill(_OOB_IDX)
        curr_to_prev_map.fill(_OOB_IDX)

        # If a request is present in both the previous and current batch,
        # we record the mapping from the prev to curr batch idx.
        req_id_to_curr_batch_idx = {
            context.request_id: curr_batch_idx
            for curr_batch_idx, context in enumerate(inputs.flat_batch)
        }

        prev_flat_batch = prev_batch.inputs.flat_batch
        for prev_idx, context in enumerate(prev_flat_batch):
            req_id = context.request_id
            # If generated_length is still 0, then there is no placeholder future
            # token. This is possible due to chunked prefill.
            if (
                req_id in req_id_to_curr_batch_idx
                and context.tokens.generated_length
            ):
                prev_to_curr_map[prev_idx] = req_id_to_curr_batch_idx[req_id]
                curr_to_prev_map[req_id_to_curr_batch_idx[req_id]] = prev_idx

        if np.all(prev_to_curr_map == _OOB_IDX):
            return None
        else:
            return prev_to_curr_map_host, curr_to_prev_map_host

    @traced
    def realize_future_tokens(
        self,
        prev_batch: AsyncBatch[TextGenerationContextType],
        inputs: TextGenerationInputs[TextGenerationContextType],
        model_inputs: ModelInputs,
    ) -> None:
        """Scatters generated tokens from the previous batch into placeholder slots.

        Fills placeholder future tokens in the current batch on the GPU.
        Returns ragged_input_tokens unchanged if there is no overlap between
        the previous and current batch.
        """
        assert isinstance(model_inputs, _HasRaggedTokens)
        mappings = self._compute_mappings(prev_batch, inputs)

        if mappings is None:
            return

        assert self._graph is not None, (
            "RealizeFutureTokenProcessor is None but there are tokens to scatter."
        )

        device = model_inputs.tokens.device

        if self._num_speculative_tokens > 0:
            assert isinstance(model_inputs, _UnifiedEagleInputs)
            assert prev_batch.spec_decode is not None
            assert model_inputs.kv_cache_inputs is not None

            num_kv_cache_inputs = len(model_inputs.kv_cache_inputs.inputs)
            cache_lengths = [
                model_inputs.kv_cache_inputs.inputs[i].cache_lengths
                for i in range(num_kv_cache_inputs)
            ]
            assert model_inputs.draft_tokens is not None
            num_draft_tokens_to_verify = model_inputs.draft_tokens.shape[1]
            num_accepted_draft_tokens = (
                prev_batch.spec_decode.num_accepted_draft_tokens_device
            )
            prev_generated_draft_tokens = (
                prev_batch.spec_decode.next_draft_tokens_device
            )
            prev_draft_tokens = (
                prev_batch.spec_decode.draft_tokens_to_verify_device
            )
            signal_buffers = getattr(model_inputs, "signal_buffers", None)

            if self._enable_dp:
                data_parallel_splits = model_inputs.data_parallel_splits
            else:
                data_parallel_splits = None
            if num_draft_tokens_to_verify == 0:
                prev_batch_size = prev_generated_draft_tokens.shape[0]
                prev_generated_draft_tokens = Buffer(
                    dtype=prev_generated_draft_tokens.dtype,
                    shape=(prev_batch_size, 0),
                    device=device,
                )

            spec_decode: (
                _RealizeFutureTokenSpecDecodeInputs[Buffer, Buffer] | None
            ) = _RealizeFutureTokenSpecDecodeInputs(
                curr_draft_tokens=model_inputs.draft_tokens,
                data_parallel_splits=data_parallel_splits,
                curr_cache_lengths=cache_lengths,
                signal_buffers=signal_buffers,
                prev_generated_draft_tokens=prev_generated_draft_tokens,
                prev_draft_tokens=prev_draft_tokens,
                prev_num_accepted_draft_tokens=num_accepted_draft_tokens,
            )
        else:
            spec_decode = None

        prev_to_curr_map, curr_to_prev_map = mappings
        device = prev_to_curr_map.device

        # TODO: This is a hotfix for MODELS-1350. Do the cleaner thing in followup.
        input_row_offsets: list[Buffer] | Buffer = (
            model_inputs.input_row_offsets
        )
        if isinstance(input_row_offsets, list):
            input_row_offsets = input_row_offsets[0]

        my_inputs = _RealizeFutureTokenInputs[Buffer, Buffer](
            prev_to_curr_map=prev_to_curr_map.to(device),
            curr_to_prev_map=curr_to_prev_map.to(device),
            curr_tokens=model_inputs.tokens,
            curr_input_row_offsets=input_row_offsets,
            prev_generated_tokens=prev_batch.generated_tokens_device,
            spec_decode=spec_decode,
        )

        out = self._graph.execute(*my_inputs.flatten())

        # Execute the realize_future_tokens kernel.
        if my_inputs.spec_decode is not None:
            assert isinstance(model_inputs, _UnifiedEagleInputs)
            (tokens, draft_tokens, *cache_lengths) = out
            model_inputs.tokens = tokens
            # This is pretty subtle. We copy the realized tokens into the original
            # draft tokens buffer so that when we read from draft_tokens later on
            # we get the real values...
            model_inputs.draft_tokens.inplace_copy_from(draft_tokens)
            assert len(model_inputs.kv_cache_inputs.inputs) == len(
                cache_lengths
            )
            for i in range(len(model_inputs.kv_cache_inputs.inputs)):
                model_inputs.kv_cache_inputs.inputs[i] = dataclasses.replace(
                    model_inputs.kv_cache_inputs.inputs[i],
                    cache_lengths=cache_lengths[i],
                )
        else:
            (new_ragged_input_tokens,) = out
            model_inputs.tokens = new_ragged_input_tokens

        # Update the model inputs with the new ragged input tokens.

        return


@final
class OverlapTextGenerationPipeline(
    TextGenerationPipelineInterface[TextGenerationContextType],
    Generic[TextGenerationContextType],
):
    """Overlap text generation pipeline."""

    _pipeline_model: PipelineModelWithKVCache[Any]

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
        disable_overlap: bool = False,
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
            disable_overlap: When this flag is set, the overlap scheduler will
                immediately synchronize after model execution. This removes any
                potential cpu / gpu overlap.

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
                f"OverlapTextGenerationPipeline requires a model with KV cache support, found {pipeline_model.__name__}"
            )

        enable_echo = self._pipeline_config.model.enable_echo
        is_spec_decode = self._pipeline_config.speculative is not None
        if is_spec_decode and enable_echo:
            raise ValueError(
                "Enable Echo is not supported for speculative decoding. Please disable echo."
            )
        elif is_spec_decode:
            return_logits = ReturnLogits.VARIABLE
        else:
            return_logits = ReturnLogits.LAST_TOKEN
        self._pipeline_model: PipelineModelWithKVCache[Any] = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            kv_cache_config=model_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=return_logits,
        )

        available_cache_memory = model_config.kv_cache._available_cache_memory
        kv_params = self._pipeline_model.kv_params

        # Load the KVCache manager.  For models with multiple KV caches
        # (e.g. sliding-window + global attention), a single manager
        # handles all caches natively via MultiKVCacheParams.
        self._spec_decode_state: SpecDecodeState | None = None
        if not is_spec_decode:
            self._kv_manager = load_kv_manager(
                params=kv_params,
                max_batch_size=self._pipeline_config.runtime.max_batch_size,
                max_seq_len=self._pipeline_model.max_seq_len,
                session=session,
                available_cache_memory=available_cache_memory,
            )
        else:
            self._spec_decode_state = SpecDecodeState.load(
                session=session,
                model=self._pipeline_model,
                pipeline_config=self._pipeline_config,
            )
            self._kv_manager = self._spec_decode_state.target_kv_manager
            if (
                self._pipeline_config.speculative is not None
                and self._pipeline_config.speculative.synthetic_acceptance_rate
                is not None
            ):
                logger.info(
                    "Synthetic acceptance rate is enabled (rate=%.2f). "
                    "Actual model acceptance will be overridden. "
                    "Results are for benchmarking only.",
                    self._pipeline_config.speculative.synthetic_acceptance_rate,
                )

        # Load sampler.
        self._sampler: Model | None = (
            session.load(
                token_sampler(
                    self._pipeline_config.sampling,
                    device=DeviceRef.from_device(self._devices[0]),
                )
            )
            if not is_spec_decode
            else None
        )

        # Overlap pipeline doesn't use structured output, so no pinned buffer
        # needed for async token transfers.
        self._pinned_new_tokens: Buffer | None = None

        # Overlap scheduling specific initialization.

        # Load the realize future tokens graph — not needed on prefill-only
        # workers (no decode phase, so no future tokens to scatter).
        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )
        self._realize_future_token_processor: (
            RealizeFutureTokenProcessor | None
        ) = (
            RealizeFutureTokenProcessor(
                session=session,
                devices=[
                    DeviceRef.from_device(device) for device in self._devices
                ],
                num_speculative_tokens=num_speculative_tokens,
                enable_dp=model_config.data_parallel_degree > 1,
            )
            if self._pipeline_config.runtime.pipeline_role
            in ("prefill_and_decode", "decode_only")
            else None
        )
        # Set previous asynchronously executing batch to None.
        self._prev_batch: AsyncBatch[TextGenerationContextType] | None = None
        self._graph_capture_runner: ServeGraphCaptureRunner | None = None
        # set a default graph capture size, 128
        self._max_graph_capture_batch_size: int = _MAX_GRAPH_CAPTURE_BATCH_SIZE

        self._disable_overlap = disable_overlap

    @property
    def _effective_max_cache_length(self) -> int:
        """Max cache length capped to the KV pool capacity."""
        return min(
            self._pipeline_model.max_seq_len,
            self._kv_manager._total_num_pages
            * self._kv_manager.params.page_size,
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Returns the tokenizer used for building contexts and decoding."""
        return self._tokenizer

    def has_pending_outputs(self) -> bool:
        """Returns True if there are pending outputs for the previous batch.

        If this is True, the caller should call ``execute()`` even with empty
        inputs to retrieve the outputs for the previous batch.
        """
        return self._prev_batch is not None

    # Warmup inputs use runtime construction with explicit max-cache-length LUT
    # sizing, so eager warmup and capture both see replay-stable buffer shapes.
    @contextmanager
    def _warmup_model_inputs(self, batch_size: int) -> Iterator[ModelInputs]:
        dp_size = self._pipeline_config.model.data_parallel_degree
        replica_batches: list[list[TextContext]] = []

        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )

        # For unified Eagle/MTP models, the graph merges prompt tokens with
        # draft tokens internally. Each request contributes 1 decode token
        # as input; the q_max_seq_len only affects KV cache dispatch metadata.
        num_decode_tokens = 1
        for _replica_idx in range(dp_size):
            replica_batches.append(
                [
                    TextContext(
                        max_length=self._pipeline_model.max_seq_len,
                        tokens=TokenBuffer(
                            np.zeros(num_decode_tokens, dtype=np.int64)
                        ),
                        eos_tracker=EOSTracker(),
                        model_name=self._pipeline_config.model.model_name,
                        _spec_decoding_state=SpecDecodingState(
                            draft_tokens_to_verify=[0] * num_speculative_tokens,
                        ),
                    )
                    for idx in range(batch_size)
                ]
            )
        with self._kv_manager.reserve(replica_batches, num_steps=1):
            max_cache_length = self._effective_max_cache_length
            kv_cache_inputs = self._kv_manager.runtime_inputs(
                replica_batches,
                num_steps=1,
                max_cache_length=max_cache_length,
            )

            return_n_logits = (
                num_speculative_tokens + 1
                if self._spec_decode_state is not None
                else 0
            )

            with Tracer("prepare_initial_token_inputs"):
                model_inputs = (
                    self._pipeline_model.prepare_initial_token_inputs(
                        replica_batches=replica_batches,
                        kv_cache_inputs=kv_cache_inputs,
                        return_n_logits=return_n_logits,
                    )
                )

            if self._spec_decode_state is not None:
                assert isinstance(model_inputs, _UnifiedEagleInputs)
                draft_tokens = Buffer.from_numpy(
                    np.zeros(
                        (batch_size * dp_size, num_speculative_tokens),
                        dtype=np.int64,
                    )
                )
                persistent_draft_tokens = (
                    self._spec_decode_state.persistent_draft_tokens
                )
                persistent_draft_tokens = _contiguous_prefix_2d(
                    persistent_draft_tokens,
                    batch_size * dp_size,
                    num_speculative_tokens,
                )
                persistent_draft_tokens.inplace_copy_from(draft_tokens)
                model_inputs.draft_tokens = persistent_draft_tokens
                model_inputs.draft_kv_blocks = (
                    self._spec_decode_state.draft_kv_blocks
                )

            yield model_inputs

    def warmup_graph_capture(self) -> None:
        """Initializes and runs overlap device graph capture warmup."""
        if not isinstance(self._pipeline_model, _SupportsModelCapture):
            raise RuntimeError(
                "Device graph capture is enabled but pipeline model does not "
                "expose a compiled model for capture/replay."
            )
        if self._pipeline_config.runtime.max_batch_size is None:
            raise RuntimeError(
                "device_graph_capture requires max_batch_size to be resolved."
            )

        max_capture_batch_size = min(
            self._pipeline_config.runtime.max_batch_size,
            _MAX_GRAPH_CAPTURE_BATCH_SIZE,
        )
        if (
            max_capture_batch_size
            < self._pipeline_config.runtime.max_batch_size
        ):
            logger.warning(
                "Capping graph capture batch size to %d "
                "(max_batch_size=%d). Decode batches above %d will fall "
                "back to eager execution.",
                max_capture_batch_size,
                self._pipeline_config.runtime.max_batch_size,
                max_capture_batch_size,
            )

        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )
        graph_capture_runner = ServeGraphCaptureRunner(
            model=self._pipeline_model.model,
            execute_model=self._pipeline_model.execute,
            session=self.session,
            kv_params=self._kv_manager.cache_params(),
            warmup_model_inputs=self._warmup_model_inputs,
            max_cache_length_upper_bound=self._effective_max_cache_length,
            max_batch_size=max_capture_batch_size,
            num_speculative_tokens=num_speculative_tokens,
            num_kv_caches=self._kv_manager.num_caches,
        )
        self._graph_capture_runner = graph_capture_runner
        self._max_graph_capture_batch_size = max_capture_batch_size
        logger.info("Starting serve device graph capture warmup.")
        graph_capture_runner.warmup_pre_ready()
        logger.info("Completed serve device graph capture warmup.")

    def _run_forward(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
        draft_tokens: Buffer | None = None,
    ) -> ModelOutputs:
        """Runs the forward pass for the provided inputs and returns the ModelOutputs.

        This handles both the non spec-decode and spec-decode paths. When running
        with spec-decode, you must provide the draft tokens even when there are
        no draft tokens to verify. In which case the shape is (batch_size, 0).
        """
        if draft_tokens is not None:
            assert self._spec_decode_state is not None
            num_draft_tokens_to_verify = draft_tokens.shape[1]
        else:
            assert self._spec_decode_state is None
            num_draft_tokens_to_verify = 0

        runner = self._graph_capture_runner
        batch_per_rank = max((len(b) for b in inputs.batches), default=0)
        use_graph_capture_replay = (
            runner is not None
            and bool(inputs)
            and inputs.batch_type == BatchType.TG
            and batch_per_rank <= self._max_graph_capture_batch_size
            and (draft_tokens is None or num_draft_tokens_to_verify > 0)
        )
        debug_verify_replay_enabled = (
            use_graph_capture_replay
            and self._pipeline_config.debug_verify_replay
        )
        debug_verify_model_inputs: ModelInputs | None = None

        # Prepare the batch.
        # Replay uses LUT buffers sized by max cache length so copied inputs
        # match captured graph buffer shapes.
        if use_graph_capture_replay:
            assert self._graph_capture_runner is not None
            with self._kv_manager.scalar_metadata_on_host():
                kv_cache_inputs = self._kv_manager.runtime_inputs(
                    inputs.batches,
                    num_steps=1,
                    max_cache_length=self._graph_capture_runner._max_cache_length_upper_bound,
                )
        else:
            kv_cache_inputs = self._kv_manager.runtime_inputs(
                inputs.batches, num_steps=1
            )

        return_n_logits = (
            num_draft_tokens_to_verify + 1 if draft_tokens is not None else 0
        )

        with Tracer("prepare_initial_token_inputs"):
            model_inputs = self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=inputs.batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )

        if debug_verify_replay_enabled:
            # Reuse non-KV buffers from replay inputs and only swap the
            # runtime-shaped KV inputs used for debug verification.
            debug_verify_model_inputs = copy.copy(model_inputs)
            debug_verify_model_inputs.update(
                kv_cache_inputs=self._kv_manager.runtime_inputs(
                    inputs.batches, num_steps=1
                )
            )

        if not isinstance(model_inputs, _HasRaggedTokens):
            raise RuntimeError(
                "OverlapTextGenerationPipeline requires model inputs with a "
                "Buffer `tokens` field."
            )
        if debug_verify_model_inputs is not None and not isinstance(
            debug_verify_model_inputs, _HasRaggedTokens
        ):
            raise RuntimeError(
                "OverlapTextGenerationPipeline requires debug-verify model "
                "inputs with a Buffer `tokens` field."
            )

        # Wrap the model inputs when speculative decoding is enabled.
        if draft_tokens is not None:
            assert self._spec_decode_state is not None
            assert isinstance(model_inputs, _UnifiedEagleInputs)
            model_inputs.draft_tokens = draft_tokens
            model_inputs.draft_kv_blocks = (
                self._spec_decode_state.draft_kv_blocks
            )

        if (
            self._prev_batch is not None
            and self._realize_future_token_processor is not None
        ):
            self._realize_future_token_processor.realize_future_tokens(
                prev_batch=self._prev_batch,
                inputs=inputs,
                model_inputs=model_inputs,
            )
            if debug_verify_model_inputs is not None:
                debug_verify_model_inputs.tokens = model_inputs.tokens

        # Execute the model and get next tokens.
        try:
            with Tracer("pipeline_model.execute"):
                if use_graph_capture_replay:
                    assert runner is not None
                    return runner.replay(
                        model_inputs=model_inputs,
                        debug_verify_replay=debug_verify_replay_enabled,
                        debug_verify_model_inputs=debug_verify_model_inputs,
                    )

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
        """Runs the forward pass, samples logits, and returns an AsyncBatch."""
        if self._spec_decode_state is not None:
            return self._execute_spec_decode(inputs)

        device0 = self._devices[0]
        assert not device0.is_host
        assert self._sampler is not None

        flat_batch = inputs.flat_batch
        with Tracer("FusedSamplingProcessor"):
            sampling_processor = FusedSamplingProcessor(
                sampler=self._sampler,
                pipeline_config=self._pipeline_config,
                context_batch=flat_batch,
                num_steps=1,
                device=device0,
                pinned_new_tokens=self._pinned_new_tokens,
            )

        model_outputs = self._run_forward(inputs)

        if model_outputs.logit_offsets is None:
            batch_size = len(flat_batch)
            logits_batch = int(model_outputs.logits.shape[0])
            if logits_batch != batch_size:
                raise AssertionError(
                    "Model returned LAST_TOKEN logits with a leading dimension "
                    f"that does not match request batch size: logits.shape[0]={logits_batch}, "
                    f"batch_size={batch_size}, input_tokens={sum(ctx.tokens.active_length for ctx in flat_batch)}, "
                    f"active_lengths={[ctx.tokens.active_length for ctx in flat_batch]}, "
                    f"generated_lengths={[ctx.tokens.generated_length for ctx in flat_batch]}."
                )

        with Tracer("apply_logits_processors"):
            apply_logits_processors(
                context_batch=flat_batch,
                batch_logits=model_outputs.logits,
                batch_logit_offsets=model_outputs.logit_offsets,
                batch_processors=[sampling_processor],
            )
        generated_tokens_device = sampling_processor.generated_tokens
        # [B, 1] -> [B]
        generated_tokens_device = generated_tokens_device.view(
            dtype=generated_tokens_device.dtype,
            shape=(generated_tokens_device.shape[0],),
        )

        # Do the copy to host for each token generated.
        with Tracer("D2H generated_tokens"):
            # Allocate a pinned tensor on the host for faster async d2h transfer
            # speeds.
            generated_tokens_host = DevicePinnedBuffer(
                shape=generated_tokens_device.shape,
                dtype=generated_tokens_device.dtype,
                device=device0,
            )
            generated_tokens_host.inplace_copy_from(generated_tokens_device)
            # Record an event to track the completion of the d2h copy.
            # This will ensure that the subsequent synchronize() call will
            # block until the d2h copy is complete, and no more.
            copy_event = device0.default_stream.record_event()

        # Make a deep copy of the input object in case the caller modifies it!
        cloned_inputs = TextGenerationInputs(
            batches=[
                [ctx for ctx in replica_batch]
                for replica_batch in inputs.batches
            ],
            num_steps=inputs.num_steps,
        )

        return AsyncBatch(
            inputs=cloned_inputs,
            generated_tokens_device=generated_tokens_device,
            generated_tokens_host=generated_tokens_host,
            copy_event=copy_event,
        )

    def _execute_spec_decode(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> AsyncBatch[TextGenerationContextType]:
        """Executes unified EAGLE speculative decoding.

        Single graph call handles: merge, target forward, greedy rejection,
        shift, and draft forward.
        """
        assert self._spec_decode_state is not None
        num_speculative_tokens = self._spec_decode_state.num_speculative_tokens

        context_batch = inputs.flat_batch
        verify_draft_tokens = all(
            ctx.tokens.generated_length > 0 for ctx in context_batch
        )
        num_draft_tokens_to_verify = (
            num_speculative_tokens if verify_draft_tokens else 0
        )

        # Delete the saved draft tokens if we are not verifying them.
        if not verify_draft_tokens:
            for ctx in context_batch:
                if len(ctx.spec_decoding_state.draft_tokens_to_verify):
                    ctx.spec_decoding_state.draft_tokens_to_verify = []

        # Load or create draft tokens.
        draft_tokens_pinned = DevicePinnedBuffer(
            shape=(len(context_batch), num_draft_tokens_to_verify),
            dtype=DType.int64,
            device=self._devices[0],
        )
        draft_tokens_np = draft_tokens_pinned.to_numpy()
        if num_draft_tokens_to_verify:
            for i, ctx in enumerate(context_batch):
                # If there are no draft_tokens to verify, populate it with a
                # arbitrary token value. This is to trigger token verification
                # more often. When we do not verify tokens, we cannot replay cuda
                # graph which hurts perf.
                if not ctx.spec_decoding_state.draft_tokens_to_verify:
                    ctx.spec_decoding_state.draft_tokens_to_verify = [
                        _MAGIC_DRAFT_TOKEN_ID
                    ] * num_draft_tokens_to_verify
                tokens = ctx.spec_decoding_state.draft_tokens_to_verify
                assert len(tokens) == num_draft_tokens_to_verify
                draft_tokens_np[i, :] = tokens

        draft_tokens_device = self._spec_decode_state.persistent_draft_tokens
        draft_tokens_device = _contiguous_prefix_2d(
            draft_tokens_device, len(context_batch), num_draft_tokens_to_verify
        )
        draft_tokens_device.inplace_copy_from(draft_tokens_pinned)

        outputs = self._run_forward(inputs, draft_tokens=draft_tokens_device)
        assert isinstance(outputs, UnifiedEagleOutputs)

        draft_tokens_pinned.inplace_copy_from(draft_tokens_device)

        # Do the copy to host for each model output using pinned memory.
        with Tracer("D2H generated_tokens"):
            device0 = self._devices[0]
            num_accepted_draft_tokens_device = outputs.num_accepted_draft_tokens
            num_accepted_draft_tokens_host = DevicePinnedBuffer(
                shape=num_accepted_draft_tokens_device.shape,
                dtype=num_accepted_draft_tokens_device.dtype,
                device=device0,
            )
            num_accepted_draft_tokens_host.inplace_copy_from(
                num_accepted_draft_tokens_device
            )

            next_tokens_device = outputs.next_tokens
            next_tokens_host = DevicePinnedBuffer(
                shape=next_tokens_device.shape,
                dtype=next_tokens_device.dtype,
                device=device0,
            )
            next_tokens_host.inplace_copy_from(next_tokens_device)

            next_draft_tokens_device = outputs.next_draft_tokens
            next_draft_tokens_host = DevicePinnedBuffer(
                shape=next_draft_tokens_device.shape,
                dtype=next_draft_tokens_device.dtype,
                device=device0,
            )
            next_draft_tokens_host.inplace_copy_from(next_draft_tokens_device)

            # Record an event to track the completion of the d2h copies.
            # This will ensure that the subsequent synchronize() call will
            # block until the d2h copy is complete, and no more.
            copy_event = device0.default_stream.record_event()

            async_batch = AsyncBatch(
                inputs=inputs,
                generated_tokens_device=next_tokens_device,
                generated_tokens_host=next_tokens_host,
                copy_event=copy_event,
                spec_decode=AsyncSpecDecodeBatch(
                    draft_tokens_to_verify_device=draft_tokens_device,
                    draft_tokens_to_verify_host=draft_tokens_pinned,
                    next_draft_tokens_device=next_draft_tokens_device,
                    next_draft_tokens_host=next_draft_tokens_host,
                    num_accepted_draft_tokens_device=num_accepted_draft_tokens_device,
                    num_accepted_draft_tokens_host=num_accepted_draft_tokens_host,
                    max_seq_len=self._pipeline_model.max_seq_len,
                ),
            )

        return async_batch

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Executes a batch of requests asynchronously on the GPU.

        This method returns before the outputs for the current batch are
        ready. The caller may need to call ``execute()`` again (possibly
        with an empty batch) to retrieve these outputs. For example:

        .. code-block:: python

            output_a = pipeline.execute(inputs)
            assert len(outputs) == 0

            output_b = pipeline.execute(empty_inputs)
            assert len(outputs) == len(inputs.flat_batch)

        Args:
            inputs: The inputs for the batch.

        Returns:
            A dictionary of request IDs to outputs. The outputs do not correspond
            to the requests in the input batch. Instead they are from the previous batch.
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
        elif self.pipeline_config.runtime.execute_empty_batches:
            # If the batch is empty and execute_empty_batches is True, we will
            # only run the forward pass to ensure that the barrier point is reached
            # for EP + DP. We skip all output processing.
            _ = self._run_forward(inputs)
            curr_batch = None
        else:
            curr_batch = None

        if self._prev_batch is not None:
            assert not self._disable_overlap, (
                "Cannot have a previous batch when overlap is disabled"
            )
            wrapped_outputs = self._prev_batch.sync_and_process_outputs()
            if self._spec_decode_state is not None:
                assert wrapped_outputs.spec_decode_metrics is not None
                self._spec_decode_state.metrics.update(
                    wrapped_outputs.spec_decode_metrics,
                )
            outputs = wrapped_outputs.output_dict

            self._prev_batch = None
        else:
            # Empty outputs as there is no previous batch.
            outputs = {}

        if curr_batch is not None:
            for context in inputs.flat_batch:
                context.update_with_future_token()
                # TODO: these two fields should not both be named spec_decode_state...
                if self._spec_decode_state is not None:
                    assert curr_batch.spec_decode is not None
                    num_draft_tokens_to_verify = (
                        curr_batch.spec_decode.num_draft_tokens_to_verify
                    )
                    context.spec_decoding_state.maybe_accepted_draft_tokens = [
                        _OOB_IDX
                    ] * num_draft_tokens_to_verify
                    if context.tokens.generated_length:
                        context.spec_decoding_state.draft_tokens_to_verify = [
                            _OOB_IDX
                        ] * self._spec_decode_state.num_speculative_tokens

        # Commit the new KV blocks into the prefix cache, ignoring the final
        # placeholder future token.
        self._kv_manager.step(inputs.batches)

        if curr_batch is not None:
            if self._disable_overlap:
                assert not outputs, (
                    "Cannot have prev outputs when overlap is disabled"
                )
                # Immediately synchronize after gpu execution and return the
                # results of the current batch.
                wrapped_outputs = curr_batch.sync_and_process_outputs()
                if self._spec_decode_state is not None:
                    assert wrapped_outputs.spec_decode_metrics is not None
                    self._spec_decode_state.metrics.update(
                        wrapped_outputs.spec_decode_metrics,
                    )
                outputs = wrapped_outputs.output_dict
            else:
                # Otherwise, delay the synchronization until the next step.
                self._prev_batch = curr_batch

        return outputs

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager.

        Note: Primary KV cache lifecycle is managed by the scheduler. This method
        handles extra KV caches managed by the pipeline model (e.g., indexer cache
        for DeepSeekV3.2).
        """
        # Primary KV cache release is handled by the scheduler via batch_constructor.
        # Pipeline model may have extra KV caches to release.
        if hasattr(self._pipeline_model, "release"):
            self._pipeline_model.release(request_id)

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        return self._kv_manager

    @property
    def draft_kv_blocks(self) -> list[Buffer] | None:
        """Returns the draft KV cache block buffers, one per DP replica.

        Returns None when speculative decoding is not active.
        """
        if self._spec_decode_state is None:
            return None
        return self._spec_decode_state.draft_kv_blocks

    def spec_decode_metrics(self) -> SpeculativeDecodingMetrics | None:
        """Returns the draft token acceptance metrics for speculative decoding."""
        if self._spec_decode_state is None:
            return None
        return self._spec_decode_state.metrics
