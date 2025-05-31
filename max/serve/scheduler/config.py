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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from max.pipelines.lib import PipelineRole

if TYPE_CHECKING:
    from max.serve.scheduler.audio_generation_scheduler import (
        AudioGenerationSchedulerConfig,
    )
from max.serve.scheduler.queues import BatchingStrategy, BatchQueueConfig


@dataclass(frozen=True)
class TokenGeneratorSchedulerConfig:
    """
    Example config

    .. code-block:: json

        {
            "context_encoding": {
                "strategy": "dynamic",
                "size": 1,
                "timeout": 0.1
            },
            "token_generation": {
                "strategy": "continuous",
                "size": 64,
                "timeout": 0.0
            }
        }
    """

    token_generation: BatchQueueConfig
    context_encoding: Optional[BatchQueueConfig] = None
    pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode
    audio_generator_scheduler_config: Optional[
        AudioGenerationSchedulerConfig
    ] = None

    @property
    def max_batch_size_tg(self) -> int:
        return self.token_generation.size

    @property
    def max_batch_size_ce(self) -> int:
        if self.context_encoding:
            return self.context_encoding.size

        return self.token_generation.size

    @property
    def max_forward_steps_tg(self) -> int:
        return self.token_generation.max_forward_steps

    @property
    def max_forward_steps_ce(self) -> int:
        if self.context_encoding:
            return self.context_encoding.max_forward_steps

        return self.token_generation.max_forward_steps

    @property
    def target_tokens_per_batch_tg(self) -> Optional[int]:
        return self.token_generation.target_sum_seq_len

    @property
    def target_tokens_per_batch_ce(self) -> Optional[int]:
        if self.context_encoding:
            return self.context_encoding.target_sum_seq_len

        return self.token_generation.target_sum_seq_len

    @property
    def enable_chunked_prefill(self) -> bool:
        return self.token_generation.enable_chunked_prefill

    @property
    def enable_in_flight_batching(self) -> bool:
        return self.token_generation.enable_in_flight_batching

    @property
    def batch_timeout(self) -> Optional[float]:
        if self.context_encoding:
            timeout = self.context_encoding.timeout
        else:
            timeout = self.token_generation.timeout

        if math.isclose(timeout, 0.0):
            return None

        return timeout

    @classmethod
    def no_cache(
        cls,
        batch_size: int,
        pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode,
    ) -> TokenGeneratorSchedulerConfig:
        """The no-cache config uses a single queue with no cache.
        Requests are dequeued into a batch and the entire batch is
        executed until all requests are completed.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            size=batch_size,
            enable_chunked_prefill=False,
        )
        config = cls(
            token_generation=token_generation_config,
            pipeline_role=pipeline_role,
        )
        return config

    @classmethod
    def continuous_heterogenous(
        cls,
        tg_batch_size: int,
        ce_batch_size: int,
        ce_batch_timeout=0.1,
        max_forward_steps=1,
        target_ce_batch_tokens=4096,
        enable_chunked_prefill: bool = True,
        enable_in_flight_batching: bool = False,
        pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode,
    ) -> TokenGeneratorSchedulerConfig:
        """The continuous-hetrogenous config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.
        """
        token_generation_config = BatchQueueConfig(
            strategy=BatchingStrategy.CONTINUOUS,
            size=tg_batch_size,
            timeout=0.0,
            max_forward_steps=max_forward_steps,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=enable_in_flight_batching,
        )
        context_encoding_config = BatchQueueConfig(
            strategy=BatchingStrategy.DYNAMIC,
            size=ce_batch_size,
            timeout=ce_batch_timeout,
            target_sum_seq_len=target_ce_batch_tokens,
        )
        config = cls(
            context_encoding=context_encoding_config,
            token_generation=token_generation_config,
            pipeline_role=pipeline_role,
        )
        return config

    @classmethod
    def paged(
        cls,
        tg_batch_size: int,
        ce_batch_size: int,
        ce_batch_timeout: float = 0.1,
        max_forward_steps: int = 1,
        target_ce_batch_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
        enable_in_flight_batching: bool = False,
        pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode,
    ) -> TokenGeneratorSchedulerConfig:
        """The paged config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.

        This config is identical to the config returned by continuous_heterogenous.
        """
        return cls.continuous_heterogenous(
            tg_batch_size=tg_batch_size,
            ce_batch_size=ce_batch_size,
            ce_batch_timeout=ce_batch_timeout,
            max_forward_steps=max_forward_steps,
            target_ce_batch_tokens=target_ce_batch_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=enable_in_flight_batching,
            pipeline_role=pipeline_role,
        )
