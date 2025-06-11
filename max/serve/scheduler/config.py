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

from dataclasses import dataclass
from typing import Optional

from max.pipelines.lib import PipelineRole


# TODO(E2EOPT-309) Delete this entire file!
@dataclass(frozen=True)
class BatchQueueConfig:
    size: int
    """Maximum number of requests in a batch."""

    max_forward_steps: int = 1
    """Maximum number of forwards steps to schedule at a time."""

    enable_chunked_prefill: bool = True
    """Enable chunked prefill to splits requests into chunks."""

    enable_in_flight_batching: bool = False
    """Enable chunked prefill to prioritize token generation requests."""

    target_sum_seq_len: Optional[int] = None
    """Target sum of the sequence lengths in the batch."""


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

    max_queue_size_tg: int | None = None
    min_batch_size_tg: int | None = None
    ce_delay_ms: float | None = None
    enable_prioritize_first_decode: bool | None = None

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
        max_forward_steps=1,
        target_ce_batch_tokens=4096,
        enable_chunked_prefill: bool = True,
        enable_in_flight_batching: bool = False,
        pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode,
        max_queue_size_tg: int | None = None,
        min_batch_size_tg: int | None = None,
        ce_delay_ms: float | None = None,
        enable_prioritize_first_decode: bool | None = None,
    ) -> TokenGeneratorSchedulerConfig:
        """The continuous-heterogenous config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.
        """
        token_generation_config = BatchQueueConfig(
            size=tg_batch_size,
            max_forward_steps=max_forward_steps,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=enable_in_flight_batching,
        )
        context_encoding_config = BatchQueueConfig(
            size=ce_batch_size,
            target_sum_seq_len=target_ce_batch_tokens,
        )
        config = cls(
            context_encoding=context_encoding_config,
            token_generation=token_generation_config,
            pipeline_role=pipeline_role,
            min_batch_size_tg=min_batch_size_tg,
            ce_delay_ms=ce_delay_ms,
            enable_prioritize_first_decode=enable_prioritize_first_decode,
            max_queue_size_tg=max_queue_size_tg,
        )
        return config

    @classmethod
    def paged(
        cls,
        tg_batch_size: int,
        ce_batch_size: int,
        max_forward_steps: int = 1,
        target_ce_batch_tokens: int = 4096,
        enable_chunked_prefill: bool = True,
        enable_in_flight_batching: bool = False,
        pipeline_role: PipelineRole = PipelineRole.PrefillAndDecode,
        max_queue_size_tg: int | None = None,
        min_batch_size_tg: int | None = None,
        ce_delay_ms: float | None = None,
        enable_prioritize_first_decode: bool | None = None,
    ) -> TokenGeneratorSchedulerConfig:
        """The paged config creates 2 queues.
        Context-encoding is done via dynamic batching.
        Token-generation is done via continuous batching.

        This config is identical to the config returned by continuous_heterogenous.
        """
        return cls.continuous_heterogenous(
            tg_batch_size=tg_batch_size,
            ce_batch_size=ce_batch_size,
            max_forward_steps=max_forward_steps,
            target_ce_batch_tokens=target_ce_batch_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_in_flight_batching=enable_in_flight_batching,
            pipeline_role=pipeline_role,
            max_queue_size_tg=max_queue_size_tg,
            min_batch_size_tg=min_batch_size_tg,
            ce_delay_ms=ce_delay_ms,
            enable_prioritize_first_decode=enable_prioritize_first_decode,
        )
