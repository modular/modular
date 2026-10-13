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
"""Input batching for Eagle3 + DeepseekV3 pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessorBase,
)
from max.pipelines.lib.interfaces import ArchConfig, BatchProcessorRuntime
from max.pipelines.lib.interfaces.batch_processor import InputsT

if TYPE_CHECKING:
    from .mha_pipeline import Eagle3MHADeepseekV3Inputs
    from .model import Eagle3DeepseekV3Inputs


class _Eagle3DeepseekV3BatchProcessorBase(
    DeepseekV3BatchProcessorBase[InputsT]
):
    """Shared Eagle3 batching: DeepseekV3 inputs plus seed and draft slot."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._seed_counter: int = 0

    def _next_seed(self) -> Buffer:
        """Returns a monotonically advancing ``uint64[1]`` seed on device 0."""
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.runtime.devices[0])


class Eagle3DeepseekV3BatchProcessor(
    _Eagle3DeepseekV3BatchProcessorBase["Eagle3DeepseekV3Inputs"]
):
    """Ragged batching for the Eagle3 + DeepseekV3 unified (MLA-draft) model."""

    def _make_mla_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        host_input_row_offsets: Buffer,
        batch_context_lengths: list[Buffer],
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: Buffer,
        data_parallel_splits: Buffer,
        ep_inputs: tuple[Buffer, ...],
    ) -> Eagle3DeepseekV3Inputs:
        from .model import Eagle3DeepseekV3Inputs

        return Eagle3DeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            draft_tokens=None,
            seed=self._next_seed(),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )


class Eagle3MHADeepseekV3BatchProcessor(
    _Eagle3DeepseekV3BatchProcessorBase["Eagle3MHADeepseekV3Inputs"]
):
    """Ragged batching for the Eagle3 MHA-draft + DeepseekV3 unified model."""

    def _make_mla_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        host_input_row_offsets: Buffer,
        batch_context_lengths: list[Buffer],
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        return_n_logits: Buffer,
        data_parallel_splits: Buffer,
        ep_inputs: tuple[Buffer, ...],
    ) -> Eagle3MHADeepseekV3Inputs:
        from .mha_pipeline import Eagle3MHADeepseekV3Inputs

        return Eagle3MHADeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=batch_context_lengths,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
            draft_tokens=None,
            seed=self._next_seed(),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
