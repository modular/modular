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

import mojo.importer
from max.driver import Device

from .kv_cache_ops import (  # type: ignore[import-not-found]
    mha_decode_num_partitions as _mha_decode_num_partitions,
)
from .kv_cache_ops import (
    mha_split_kv_decode_num_partitions as _mha_split_kv_decode_num_partitions,
)
from .kv_cache_ops import (
    mla_dispatch_args_scalar as _mla_dispatch_args_scalar,
)
from .kv_cache_ops import (
    multiprocessor_count as _multiprocessor_count,
)


def mha_decode_num_partitions(
    batch_size: int,
    max_cache_valid_length: int,
    n_kv_heads: int,
    device: Device,
) -> int:
    """Returns the MHA decode partition count for the given request."""
    return int(
        _mha_decode_num_partitions(
            batch_size,
            max_cache_valid_length,
            n_kv_heads,
            device._device_context_ptr(),
        )
    )


def mha_split_kv_decode_num_partitions(
    batch_size: int,
    num_keys: int,
    kv_heads_per_device: int,
    sliding_window: int | None,
    sm_count: int,
) -> int:
    """Returns the decode partition count for a KV cache.

    The count covers a cache whose key is wider than its value.

    The SM100 kernel honors this count from the dispatch metadata, so an
    engine that keys graph captures on it must compute the same value.

    Args:
        batch_size: Requests in the batch.
        num_keys: The batch's largest cache length.
        kv_heads_per_device: KV heads on one device.
        sliding_window: The group's attention window, or ``None``.
        sm_count: The device's multiprocessor count.
    """
    return int(
        _mha_split_kv_decode_num_partitions(
            batch_size,
            num_keys,
            kv_heads_per_device,
            sliding_window or 0,
            sm_count,
        )
    )


def multiprocessor_count(device: Device) -> int:
    """Returns the device's multiprocessor (SM) count."""
    return int(_multiprocessor_count(device._device_context_ptr()))


def mla_dispatch_args_scalar(
    batch_size: int,
    max_cache_valid_length: int,
    q_max_seq_len: int,
    num_heads: int,
    is_fp8_kv: bool,
    device: Device,
    sparse_indices_stride: int = 0,
) -> tuple[int, int, int]:
    """Returns the MLA dispatch metadata scalars for the given request.

    Returns ``(batch_size, q_max_seq_len, num_partitions)``.
    These three values populate the size-3 device-side buffer that the
    kernel reads as ``scalar_args``.  ``effective_split_len`` is no longer
    returned; the MoGG ops compute it directly from ``max_cache_length``.

    ``sparse_indices_stride`` is the selected rows per token of a sparse
    MLA cache (``0`` for dense). On AMD it sizes ``num_partitions`` from the
    selection instead of from ``max_cache_valid_length``, matching the
    split the sparse decode kernel launches.
    """
    result = _mla_dispatch_args_scalar(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        device._device_context_ptr(),
        sparse_indices_stride,
    )
    return int(result[0]), int(result[1]), int(result[2])


__all__ = [
    "mha_decode_num_partitions",
    "mha_split_kv_decode_num_partitions",
    "mla_dispatch_args_scalar",
    "multiprocessor_count",
]
