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
"""Implements the DeepseekV3.2 model."""

from __future__ import annotations

from typing import Any

from max.graph import BufferType, BufferValue, TensorType, TensorValue, Value
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import Module

from .model_config import DeepseekV3_2Config


class DeepseekV3_2(Module):
    """Defines the DeepseekV3.2 transformer model."""

    def __init__(self, config: DeepseekV3_2Config) -> None:
        raise NotImplementedError

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
    ) -> tuple[TensorValue, ...]:
        raise NotImplementedError

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        raise NotImplementedError
