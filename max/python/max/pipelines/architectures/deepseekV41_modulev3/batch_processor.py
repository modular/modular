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
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputs
from max.nn.kv_cache.cache_params import KVCacheParamInterface
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
    single_replica_context_batch,
)

from .inputs import DeepseekV41Inputs, v41_config
from .interface import extra_input_names, extra_inputs

# Engram hashes the 4-gram ending at each token, so a chunk needs the three
# tokens before it.
_NUM_LOOKBACK = 3


class DeepseekV41BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, DeepseekV41Inputs]
):
    """Ragged batching for DeepSeek-V4.1-Flash, with the Engram and resume inputs.

    The inputs after the KV leaves are produced in
    :func:`~.interface.extra_input_names` order, so the compile ABI is read
    from one place. The Engram tables are static: the model binds them
    after loading with :meth:`bind_engram`.
    """

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._v41 = v41_config(runtime.pipeline_config.model.huggingface_config)
        self._engram_static: dict[str, Buffer] = {}
        if runtime.pipeline_config.model.data_parallel_degree != 1:
            raise ValueError(
                "DeepSeek-V4.1 ModuleV3 currently requires data_parallel_degree=1"
            )

    def bind_engram(self, static_inputs: Mapping[str, Buffer]) -> None:
        """Hands over the ``engram_table_*`` buffers the model pinned."""
        self._engram_static = dict(static_inputs)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns the compile inputs up to the signal buffers, in ABI order."""
        inputs: list[TensorType | BufferType] = list(
            modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params, device_refs=device_refs
            )
        )
        inputs.extend(t for _, t in extra_inputs(self._v41, device_refs))
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
                "tables must be bound before the first batch."
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
        """Packs one ragged batch with its KV, Engram and replay inputs."""
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        assert kv_cache_inputs is not None
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        device0 = self.runtime.devices[0]
        return DeepseekV41Inputs(
            tokens=Buffer.from_numpy(tokens_np.astype(np.int64)).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            extras=self._extras(context_batch, offsets_np),
        )
