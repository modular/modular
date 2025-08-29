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

from collections.abc import Iterator
from dataclasses import dataclass, fields

from max.graph import TensorValue
from max.nn.kv_cache.cache_params import KVCacheParams

from .attention.mask_config import MHAMaskVariant


@dataclass
class PagedKVCacheTensorsNoOpaque:
    blocks: TensorValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    is_cache_empty: TensorValue

    def __iter__(self) -> Iterator[TensorValue]:
        for field in fields(self):
            yield getattr(self, field.name)


def rope_no_opaque(
    input: TensorValue,
    input_row_offsets: TensorValue,
    start_pos: TensorValue,
    freqs_cis: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    # TODO: implement
    raise NotImplementedError("rope_no_opaque not implemented")


def store_k_cache(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_k: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    # TODO: implement
    raise NotImplementedError("store_k_cache not implemented")


def store_v_cache(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_v: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    # TODO: implement
    raise NotImplementedError("store_v_cache not implemented")


def flash_attention_ragged_no_opaque(
    kv_params: KVCacheParams,
    input: TensorValue,
    layer_idx: TensorValue,
    kv_collection: PagedKVCacheTensorsNoOpaque,
    input_row_offsets: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float = 1.0,
) -> TensorValue:
    # TODO: implement
    raise NotImplementedError(
        "flash_attention_ragged_no_opaque not implemented"
    )
