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

"""Naive KV cache for the Transformer."""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, TypeVar, cast

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, TensorType

from .cache_params import KVCacheParams
from .context import KVCacheAwareContext
from .manager import (
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    PaddedKVCacheInputs,
)

T = TypeVar("T", bound=KVCacheAwareContext)


@dataclass
class NaiveKVCacheInputSymbols(KVCacheInputSymbols):
    k_cache: BufferType
    v_cache: BufferType
    start_pos: TensorType
    null_op: TensorType


class NaiveKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
        session: InferenceSession,
    ) -> None:
        assert len(devices) == 1, "Naive caching only supports a single device."
        assert params.n_devices == 1, (
            "Naive caching only supports a single device."
        )
        if params.enable_prefix_caching:
            raise ValueError("Prefix caching is not supported for naive cache.")
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=False,
        )

        self.keys = Tensor(
            shape=self.cache_shape,
            dtype=self.params.dtype,
            device=self.devices[0],
        )

        self.values = Tensor(
            shape=self.cache_shape,
            dtype=self.params.dtype,
            device=self.devices[0],
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        return (
            reduce(
                mul,
                cls._cache_shape(
                    params, max_batch_size, max_seq_len, num_layers
                ),
            )
            * params.dtype.size_in_bytes
            * 2
        )

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        cache_size_per_sequence = (
            reduce(
                mul,
                cls._cache_shape(params, 1, max_seq_len, num_layers),
            )
            * params.dtype.size_in_bytes
            * 2
        )
        return int(available_cache_memory // cache_size_per_sequence)

    @property
    def cache_shape(self) -> list[int]:
        return self._cache_shape(
            self.params,
            self.max_batch_size,
            self.max_seq_len,
            self.num_layers,
        )

    @staticmethod
    def _cache_shape(
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
    ):
        return [
            max_seq_len,
            num_layers,
            max_batch_size,
            params.n_kv_heads,
            params.head_dim,
        ]

    def fetch(
        self,
        batch: list[T],
        num_steps: int = 1,
    ) -> list[KVCacheInputs]:
        existing_keys = list(self.active)
        for i, ctx in enumerate(batch):
            seq_id = ctx.cache_seq_id
            prompt = ctx.next_tokens
            if existing_keys[i] != seq_id:
                msg = (
                    "seq_ids passed, are different than current inflight"
                    " batch.Naive Caching currently does not support mutating"
                    " inflight batches."
                )
                raise ValueError(msg)

            total_length = ctx.start_idx + len(prompt) + num_steps - 1
            assert total_length <= self.max_seq_len, (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens and {num_steps} steps. Existing length: {ctx.start_idx}"
            )

        max_cache_len = max(ctx.start_idx for ctx in batch)
        padded_kv_cache_inputs = [
            PaddedKVCacheInputs(
                k_cache=self.keys,
                v_cache=self.values,
                start_pos=Tensor.scalar(
                    max_cache_len, DType.int64, self.devices[0]
                ),
                # TODO: MSDK-1201 - This next variable is not used upstream.
                # It is included here, as a placeholder, until we can dynamically
                # return a number of tensors from both `fetch` and `input_symbols`.
                null_op=Tensor.scalar(
                    max_cache_len, DType.int64, self.devices[0]
                ),
            )
        ]
        return cast(list[KVCacheInputs], padded_kv_cache_inputs)

    def input_symbols(
        self,
    ) -> list[NaiveKVCacheInputSymbols]:
        return [
            NaiveKVCacheInputSymbols(
                k_cache=BufferType(
                    self.params.dtype,
                    shape=[
                        "max_seq_len",
                        self.num_layers,
                        "max_batch_size",
                        self.params.n_kv_heads,
                        self.params.head_dim,
                    ],
                ),
                v_cache=BufferType(
                    self.params.dtype,
                    shape=[
                        "max_seq_len",
                        self.num_layers,
                        "max_batch_size",
                        self.params.n_kv_heads,
                        self.params.head_dim,
                    ],
                ),
                start_pos=TensorType(DType.int64, shape=[]),
                # null_op - this isnt used for the naive cache
                null_op=TensorType(DType.int64, shape=[]),
            )
        ]
