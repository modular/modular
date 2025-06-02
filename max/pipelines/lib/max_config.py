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
"""MAX config classes."""

from __future__ import annotations

import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from max.dtype import DType
from max.engine import GPUProfilingMode
from max.nn.kv_cache import KVCacheStrategy

logger = logging.getLogger("max.pipelines")


@dataclass
class MAXConfig:
    """Abstract base class for all MAX configs.

    There are some invariants that :obj:`MAXConfig` classes should follow:
    - All config classes should be dataclasses.
    - All config classes should have a :obj:`help()` method that returns a dictionary of config
    options and their descriptions.
    - All config classes dataclass fields should have default values, and hence
    can be trivially initialized via :obj:`cls()`.
    - All config classes should be frozen (except :obj:`KVCacheConfig` for now), to
    avoid accidental modification of config objects.
    - All config classes must have mutually exclusive dataclass fields among
    themselves.
    """

    @abstractmethod
    def help(self) -> dict[str, str]:
        """Documentation for this config class. Return a dictionary of config
        options and their descriptions."""
        ...


# frozen is False (for now) because of _available_cache_memory being set by
# internal code.
@dataclass(frozen=False)
class KVCacheConfig(MAXConfig):
    cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT
    """The cache strategy to use. This defaults to :obj:`model_default`, which will set the cache
    strategy based on the default strategy for the architecture requested.

    You can also force the engine to use a specific caching strategy: :obj:`continuous` | :obj:`paged`.
    """

    kv_cache_page_size: int = 128
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted."""

    device_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace. The calculation is:

    .. math::

        kv\\_cache\\_workspace = (total\\_free\\_memory \\times device\\_memory\\_utilization) - model\\_weights\\_size
    """

    host_kvcache_swap_space_gb: float = 50.0
    """The amount of host memory to use for the host KVCache in GiB.

    This space is only allocated when kvcache_swapping_to_host is enabled.
    """

    _available_cache_memory: Optional[int] = None
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "cache_strategy": "Force a specific cache strategy: 'paged' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 512.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "enable_kvcache_swapping_to_host": "Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted. This defaults to false.",
            "device_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size. Default is set to 0.9.",
            "host_kvcache_swap_space_gb": "The amount of host memory to use for the host KVCache in GiB. This is only used when kvcache_swapping_to_host is enabled. Default is set to 50.0.",
        }


@dataclass
class SamplingConfig(MAXConfig):
    top_k: int = 1
    """Limits the sampling to the K most probable tokens. This defaults to 1, which enables greedy sampling."""

    top_p: float = 1
    """Only use the tokens whose cumulative probability within the top_p threshold. This applies to the top_k tokens."""

    temperature: float = 1
    """Controls the randomness of the model's output; higher values produce more diverse responses."""

    in_dtype: DType = DType.float32
    """The data type of the input tokens."""

    out_dtype: DType = DType.float32
    """The data type of the output logits."""

    frequency_penalty: float = 0.0
    """The frequency penalty to apply to the model's output. A positive value will penalize new tokens
    based on their frequency in the generated text: tokens will receive a penalty proportional to the
    count of appearances."""

    presence_penalty: float = 0.0
    """The presence penalty to apply to the model's output. A positive value will penalize new tokens
    that have already appeared in the generated text at least once by applying a constant penalty."""

    repetition_penalty: float = 1.0
    """The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens
    that have already appeared in prompt and generated text at least once by dividing the logits by the
    repetition penalty."""

    seed: int = 0
    """The seed to use for the random number generator."""

    enable_structured_output: bool = False
    """Enable structured generation/guided decoding for the server. This allows the user to pass a json
    schema in the response_format field, which the LLM will adhere to."""

    enable_variable_logits: bool = False
    """Enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with
    their associated logit_offsets. This is needed to produce additional logits for echo and speculative
    decoding purposes."""

    do_penalties: bool = False
    """Whether to apply frequency and presence penalties to the model's output."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "top_k": "Limit sampling to the top K most probable tokens during generation. This can help control randomness and improve output quality. This defaults to 1, which defaults to greedy sampling.",
            "top_p": "Only use the tokens whose cumulative probability within the top_p threshold. This applies to the top_k tokens.",
            "temperature": "Controls the randomness of the model's output; higher values produce more diverse responses.",
            "frequency_penalty": "The frequency penalty to apply to the model's output. A positive value will penalize new tokens based on their frequency in the generated text: tokens will receive a penalty proportional to the count of appearances.",
            "presence_penalty": "The presence penalty to apply to the model's output. A positive value will penalize new tokens that have already appeared in the generated text at least once by applying a constant penalty.",
            "repetition_penalty": "The repetition penalty to apply to the model's output. Values > 1 will penalize new tokens that have already appeared in prompt and generated text at least once by dividing the logits by the repetition penalty.",
            "seed": "The seed to use for the random number generator. This defaults to 0.",
            "enable_structured_output": "Whether to enable constrained decoding in the text generation pipeline. This defaults to false.",
        }


@dataclass
class ProfilingConfig(MAXConfig):
    gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF
    """Whether to enable GPU profiling of the model."""

    def __post_init__(self):
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == GPUProfilingMode.OFF:
            try:
                self.gpu_profiling = GPUProfilingMode(gpu_profiling_env)
            except ValueError:
                valid_values = [mode.value for mode in GPUProfilingMode]
                raise ValueError(
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "gpu_profiling": "Whether to turn on GPU profiling for the model. This defaults to 'off'.",
        }
