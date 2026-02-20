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

from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.lib import PipelineConfig


@dataclass
class TokenGenerationSchedulerConfig:
    """Scheduler configuration."""

    max_batch_size: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""

    target_tokens_per_batch_ce: int
    """The target total number of tokens to encode in the context encoding batch."""

    max_seq_len: int | None = None
    """The maximum sequence length of the model."""

    max_batch_total_tokens: int | None = None
    """Ensures the sum of page-aligned context lengths in a batch does not
    exceed max_batch_total_tokens. Alignment uses the KV cache page size."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context encoding requests."""

    data_parallel_degree: int = 1
    """Data-parallelism parameter. The degree to which the model is replicated
    is dependent on the model type."""

    kvcache_ce_watermark: float = 0.95
    """The maximum percentage of total KVCache memory that can be used after allocating a CE request. This parameter was found empirically."""

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError(
                f"`max_batch_size` must be greater than 0, found {self.max_batch_size}"
            )
        if self.target_tokens_per_batch_ce <= 0:
            raise ValueError(
                f"`target_tokens_per_batch_ce` must be greater than 0, found {self.target_tokens_per_batch_ce}"
            )
        if (
            self.enable_chunked_prefill
            and self.target_tokens_per_batch_ce is None
        ):
            raise ValueError(
                "Need set `target_tokens_per_batch_ce` for the scheduler to enable chunked prefill."
            )
        if self.max_forward_steps_tg <= 0:
            raise ValueError(
                f"`max_forward_steps_tg` must be greater than 0, found {self.max_forward_steps_tg}"
            )
        if (
            self.max_batch_total_tokens is not None
            and self.max_seq_len is not None
            and self.max_batch_total_tokens < self.max_seq_len
        ):
            raise ValueError(
                f"`max_batch_total_tokens` must be greater than or equal to `max_seq_len`, found {self.max_batch_total_tokens} < {self.max_seq_len}"
            )
        if self.max_batch_size > self.target_tokens_per_batch_ce:
            raise ValueError(
                f"`max_batch_size` must be less than or equal to `target_tokens_per_batch_ce`, found {self.max_batch_size} > {self.target_tokens_per_batch_ce}"
            )

    @classmethod
    def from_pipeline_config(
        cls, pipeline_config: PipelineConfig
    ) -> TokenGenerationSchedulerConfig:
        # We know that the max_length and max_batch_size is not None since they
        # are required for memory estimation.
        assert pipeline_config.max_batch_size is not None

        return cls(
            max_batch_size=pipeline_config.max_batch_size,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            target_tokens_per_batch_ce=pipeline_config.max_batch_input_tokens,
            max_seq_len=pipeline_config.model.max_length,
            max_batch_total_tokens=pipeline_config.max_batch_total_tokens,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            kvcache_ce_watermark=pipeline_config.kvcache_ce_watermark,
        )
