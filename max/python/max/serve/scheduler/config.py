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

from max.pipelines.lib import PipelineConfig


@dataclass
class TokenGenerationSchedulerConfig:
    """Scheduler configuration."""

    max_batch_size_tg: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""

    max_batch_size_ce: int
    """The maximum number of requests that can be in the context encoding batch."""

    target_tokens_per_batch_ce: int = 8192
    """The target total number of tokens to encode in the context encoding batch."""

    enable_chunked_prefill: bool = True
    """Enables chunked prefill, where the scheduler splits requests into chunks to ensure
    each batch contains exactly `target_tokens_per_batch_ce` tokens."""

    enable_in_flight_batching: bool = False
    """When enabled, prioritizes token generation by batching it with context encoding requests."""

    data_parallel_degree: int = 1
    """Data-parallelism parameter. The degree to which the model is replicated
    is dependent on the model type."""

    def __post_init__(self) -> None:
        if self.max_batch_size_tg <= 0:
            raise ValueError(
                f"`max_batch_size_tg` must be greater than 0, found {self.max_batch_size_tg}"
            )
        if self.max_batch_size_ce <= 0:
            raise ValueError(
                f"`max_batch_size_ce` must be greater than 0, found {self.max_batch_size_ce}"
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

    @classmethod
    def from_pipeline_config(
        cls, pipeline_config: PipelineConfig
    ) -> TokenGenerationSchedulerConfig:
        return cls(
            max_batch_size_tg=pipeline_config.max_batch_size
            if pipeline_config.max_batch_size is not None
            else 1,
            max_forward_steps_tg=pipeline_config.max_num_steps
            if pipeline_config.max_num_steps != -1
            else 1,
            max_batch_size_ce=pipeline_config.max_ce_batch_size,
            target_tokens_per_batch_ce=pipeline_config.prefill_chunk_size,
            enable_chunked_prefill=pipeline_config.enable_chunked_prefill,
            enable_in_flight_batching=pipeline_config.enable_in_flight_batching,
            data_parallel_degree=pipeline_config.model_config.data_parallel_degree,
        )
