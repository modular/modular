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
"""Input batching for the DeepSeek-V4.1-Flash ModuleV3 pipeline model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPCommInitializer
from max.nn.legacy.kv_cache import KVCacheInputs
from max.nn.legacy.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    engram_lookback_token_ids,
)
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
    RaggableContext,
    build_single_replica_ragged_token_arrays,
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.utils import compute_data_parallel_splits

from .inputs import DeepseekV41Inputs, v41_config
from .interface import extra_input_names, extra_inputs

# Engram hashes the 4-gram ending at each token, so a chunk needs the three
# tokens before it.
_NUM_LOOKBACK = 3


def decoder_row_count(row_offsets: np.ndarray, window: int) -> int:
    """Rows the decoder layers run: each request's last ``window`` rows."""
    return int(np.minimum(np.diff(row_offsets.astype(np.int64)), window).sum())


class DeepseekV41BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, DeepseekV41Inputs]
):
    """Ragged batching for DeepSeek-V4.1-Flash, with the Engram and resume inputs.

    The inputs after the KV leaves are produced in
    :func:`~.interface.extra_input_names` order, so the compile ABI is read
    from one place. The Engram tables and token map are static: the model binds
    them after loading with :meth:`bind_engram`.
    """

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._v41 = v41_config(runtime.pipeline_config.model.huggingface_config)
        self._engram_static: dict[str, Buffer] = {}
        self._dp_degree = runtime.pipeline_config.model.data_parallel_degree
        self._ep_comm_initializer: EPCommInitializer | None = None

    def bind_engram(self, static_inputs: Mapping[str, Buffer]) -> None:
        """Hands over the ``engram_table_*`` and ``engram_token_map`` buffers."""
        self._engram_static = dict(static_inputs)

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires the EP buffers the model's load created, if it did."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        """The EP comm buffers, in ``EPBatchManager.input_types`` order."""
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns the compile inputs up to the signal buffers, in ABI order.

        Under data parallelism the CPU split boundaries and int64 row
        offsets follow ``input_row_offsets`` and precede the KV leaves,
        matching ``DeepseekV41.forward``'s variadic order (the DeepseekV3
        ModuleV3 pattern).
        """
        inputs: list[TensorType | BufferType] = list(
            modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params, device_refs=device_refs
            )
        )
        if self._dp_degree > 1:
            inputs.insert(
                3,
                TensorType(
                    DType.int64,
                    shape=[self._dp_degree + 1],
                    device=DeviceRef.CPU(),
                ),
            )
            inputs.insert(
                4,
                TensorType(
                    DType.int64,
                    shape=["input_row_offsets_len"],
                    device=DeviceRef.CPU(),
                ),
            )
        inputs.extend(
            t for _, t in extra_inputs(self._v41, device_refs, self._dp_degree)
        )
        return inputs

    def _extras(
        self,
        context_batch: Sequence[TextContext],
        row_offsets: np.ndarray,
    ) -> tuple[Buffer, ...]:
        device0 = self.runtime.devices[0]
        batch = len(context_batch)
        total_seq_len = int(row_offsets[-1])
        per_request: dict[str, np.ndarray] = {
            # No request resumes yet: the window starts at the beginning of
            # every request and nothing is below the write floor. A request
            # resumed from cached pages needs its real window start and
            # write floor here; the registration keeps prefix caching off
            # until that is supported.
            "swa_begin": np.zeros(batch, np.int32),
            "global_write_floor": np.zeros(batch, np.int32),
            # Text only: no position is an image token.
            "image_token_mask": np.zeros(total_seq_len, np.uint8),
            "engram_lookback": engram_lookback_token_ids(
                [ctx.tokens.all for ctx in context_batch],
                [ctx.tokens.processed_length for ctx in context_batch],
                _NUM_LOOKBACK,
            ).astype(np.int32),
            "engram_lookback_dead": np.zeros((batch, _NUM_LOOKBACK), np.uint8),
            # The graph reads only the extent: the rows the decoder layers run.
            "decoder_rows": np.zeros(
                decoder_row_count(row_offsets, self._v41.sliding_window),
                np.uint8,
            ),
        }
        buffers = {
            name: Buffer.from_numpy(array).to(device0)
            for name, array in per_request.items()
        }
        names = extra_input_names(self._v41)
        missing = [
            n
            for n in names
            if n not in buffers and n not in self._engram_static
        ]
        if missing:
            raise RuntimeError(
                f"no value for the DeepSeek-V4.1 inputs {missing}; the Engram "
                "tables and token map must be bound before the first batch."
            )
        return tuple(
            buffers[name] if name in buffers else self._engram_static[name]
            for name in names
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV41Inputs:
        """Packs one ragged batch with its KV and Engram inputs.

        Under data parallelism all replicas' requests form one device-0
        batch plus the CPU ``data_parallel_splits`` boundaries the forward
        splits it by.
        """
        if self._dp_degree > 1 and len(replica_batches) != self._dp_degree:
            raise ValueError(
                f"data parallelism expects {self._dp_degree} replica "
                f"batches, got {len(replica_batches)}."
            )
        assert kv_cache_inputs is not None
        # All replicas pack as one device-0 batch (the deepseekV3 ModuleV3
        # DP_EP pattern). An empty replica contributes a zero-width
        # boundary; the in-graph split still hands it a 0-token shard.
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        if context_batch:
            tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
                cast(Sequence[RaggableContext], context_batch)
            )
        else:
            tokens_np = np.empty(0, dtype=np.int64)
            offsets_np = np.zeros(1, dtype=np.uint32)
        device0 = self.runtime.devices[0]
        data_parallel_splits = None
        input_row_offsets_i64 = None
        if self._dp_degree > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
            input_row_offsets_i64 = Buffer.from_numpy(
                offsets_np.astype(np.int64)
            )
        return DeepseekV41Inputs(
            tokens=Buffer.from_numpy(tokens_np.astype(np.int64)).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            extras=self._extras(context_batch, offsets_np),
            ep_inputs=self._ep_inputs(),
            data_parallel_splits=data_parallel_splits,
            input_row_offsets_i64=input_row_offsets_i64,
        )
