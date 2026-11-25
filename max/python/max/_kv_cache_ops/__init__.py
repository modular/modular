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

from dataclasses import dataclass
from enum import IntEnum, IntFlag

import mojo.importer
from max.driver import Device

from .kv_cache_ops import (  # type: ignore[import-not-found]
    attn_dispatch_resolve as _attn_dispatch_resolve,
)
from .kv_cache_ops import (
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

ATTN_DISPATCH_UNSET = -1
"""Value of a descriptor field its attention family does not set.

Zero is a valid window and a valid stride, so it cannot mean "unset".
"""


class AttnKVDType(IntEnum):
    """The KV cache storage classes a decode dispatch tells apart."""

    OTHER = 0
    FP8 = 1


class AttnDispatchFlag(IntFlag):
    """Yes-or-no facts of a KV cache that a decode dispatch reads."""

    NONE = 0
    SPLIT_HEAD_DIMS = 1 << 0
    """The cache's key is wider than its value."""
    FIXED_TOPK = 1 << 1
    """A sparse decode passes neither per-token top-k lengths nor a sink."""
    PHYSICAL_INDICES = 1 << 2
    """Sparse rows are physical."""
    ROW_INVARIANT_VERIFY = 1 << 3
    """The launch is the row-invariant pin's own verify."""


@dataclass(frozen=True, kw_only=True)
class AttnDispatchDesc:
    """The facts of one KV cache that its decode dispatch policy reads.

    The fields are the union over every attention family. A policy ignores
    the fields it does not read and refuses a field it needs that is
    :data:`ATTN_DISPATCH_UNSET`. What the device knows, its API and its
    multiprocessor count, is not here: the policy reads it from the device.
    """

    kind: str
    """The attention family: ``"mha"``, ``"mla"``, ``"msa"``, ``"msa_index"``."""
    num_q_heads: int = ATTN_DISPATCH_UNSET
    """Query heads on one device."""
    num_kv_heads: int = ATTN_DISPATCH_UNSET
    """KV heads on one device."""
    head_dim: int = ATTN_DISPATCH_UNSET
    """The cache row width the dispatch assumes."""
    page_size: int = ATTN_DISPATCH_UNSET
    window_size: int = ATTN_DISPATCH_UNSET
    """The attention window, unset when there is none."""
    sparse_indices_stride: int = ATTN_DISPATCH_UNSET
    """The sparse index row stride, unset for a dense cache."""
    kv_dtype: int = ATTN_DISPATCH_UNSET
    """An :class:`AttnKVDType` value."""
    kv_form: int = ATTN_DISPATCH_UNSET
    """The sparse MLA cache form, an :class:`MLASparseKVForm` value."""
    flags: AttnDispatchFlag = AttnDispatchFlag.NONE

    def int_fields(self) -> list[int]:
        """Returns the integer fields in the order the binding reads them."""
        return [
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            self.window_size,
            self.sparse_indices_stride,
            self.kv_dtype,
            self.kv_form,
            int(self.flags),
        ]


@dataclass(frozen=True)
class AttnDispatch:
    """What one decode dispatch resolves to."""

    key: tuple[int, ...]
    """The capture identity. Compare keys for equality only."""
    metadata: tuple[int, ...]
    """The scalars of the dispatch record, empty when no kernel reads one."""


def attn_dispatch_resolve(
    device: Device,
    desc: AttnDispatchDesc,
    batch_size: int,
    q_max_seq_len: int,
    max_cache_valid_length: int,
) -> AttnDispatch:
    """Resolves the decode dispatch of one KV cache at one step shape.

    Every attention family resolves here, through the policy the kernels
    register for the descriptor's kind, so the result follows the launch.

    Args:
        device: The device the decode runs on.
        desc: The cache's dispatch descriptor.
        batch_size: Requests in the decode batch.
        q_max_seq_len: Query rows per request: ``1`` for plain decode.
        max_cache_valid_length: The batch's largest valid cache length.

    Returns:
        The capture key and the dispatch record scalars.

    Raises:
        Exception: If the kind is unknown, the shape is invalid, a field
            the kind's policy needs is unset, or the kernels do not support
            the shape.
    """
    resolved = [
        int(value)
        for value in _attn_dispatch_resolve(
            device._device_context_ptr(),
            desc.kind,
            desc.int_fields(),
            batch_size,
            q_max_seq_len,
            max_cache_valid_length,
        )
    ]
    n_key = resolved[0]
    return AttnDispatch(
        key=tuple(resolved[1 : 1 + n_key]),
        metadata=tuple(resolved[1 + n_key :]),
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
    row_invariant_verify: bool = False,
) -> tuple[int, int, int]:
    """Returns the MLA dispatch metadata scalars for the given request.

    Returns ``(batch_size, q_max_seq_len, num_partitions)``.
    These three values populate the size-3 device-side buffer that the
    kernel reads as ``scalar_args``.  ``effective_split_len`` is no longer
    returned; the MoGG ops compute it directly from ``max_cache_length``.

    ``sparse_indices_stride`` is the selected rows per token of a sparse
    MLA cache (``0`` for dense). On AMD it sizes ``num_partitions`` from the
    selection instead of from ``max_cache_valid_length``, matching the
    split the sparse decode kernel launches. On NVIDIA it follows the
    SM100 sparse launch at head counts 8 through 64 in multiples of eight
    and at 128 heads.

    ``row_invariant_verify`` says the launch this key describes is the
    row-invariant pin's own verify, which sizes its split from the shape
    rather than the live cache; the key has to follow the launch.
    """
    key = mla_dispatch_key(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        device,
        sparse_indices_stride,
        row_invariant_verify=row_invariant_verify,
    )
    return key[0], key[1], key[2]


class MLASparseKVForm(IntEnum):
    """The cache forms the SM100 sparse MLA dispatch key distinguishes."""

    PLAIN = 0
    """A sparse MLA cache without scales: unit-scale FP8 or BF16."""
    SCALED_FP8 = 1
    """A sparse MLA cache of FP8 with block scales."""
    NVFP4 = 2
    """A packed NVFP4 sparse MLA cache."""


def mla_dispatch_key(
    batch_size: int,
    max_cache_valid_length: int,
    q_max_seq_len: int,
    num_heads: int,
    is_fp8_kv: bool,
    device: Device,
    sparse_indices_stride: int = 0,
    *,
    kv_form: MLASparseKVForm = MLASparseKVForm.PLAIN,
    page_size: int = 128,
    fixed_topk: bool = True,
    physical_indices: bool = False,
    depth: int = 0,
    row_invariant_verify: bool = False,
) -> tuple[int, int, int, int]:
    """Returns :func:`mla_dispatch_args_scalar`'s scalars and a kernel variant.

    The variant distinguishes 64-key single-CTA (``0``), 128-key paired
    (``1``), and 128-key single-CTA (``2``) launches. It is part of the
    capture identity; device dispatch metadata still contains three scalars.

    SM100 sparse keys follow the launched split and variant at head counts
    8 through 64 in multiples of eight and at 128 heads. The selection also
    depends on ``kv_form``, ``page_size``, and ``fixed_topk``: whether the
    decode passes neither per-token top-k lengths nor an attention sink.
    ``physical_indices`` identifies rows already mapped to cache storage,
    allowing policies measured for that form. It defaults to logical indices;
    NVFP4 always uses physical value and scale rows regardless of this flag.
    ``depth`` is the cache row width; its default of zero keeps policies that
    require a measured row width disabled.
    ``row_invariant_verify`` says the launch this key describes is the
    row-invariant pin's own verify: its split comes from the shape, not the
    live cache, so the key has to take the same count or a captured graph
    replays a launch the step does not make.
    """
    result = _mla_dispatch_args_scalar(
        batch_size,
        max_cache_valid_length,
        q_max_seq_len,
        num_heads,
        is_fp8_kv,
        device._device_context_ptr(),
        sparse_indices_stride,
        kv_form,
        page_size,
        fixed_topk,
        physical_indices,
        depth,
        row_invariant_verify,
    )
    return int(result[0]), int(result[1]), int(result[2]), int(result[3])


__all__ = [
    "ATTN_DISPATCH_UNSET",
    "AttnDispatch",
    "AttnDispatchDesc",
    "AttnDispatchFlag",
    "AttnKVDType",
    "MLASparseKVForm",
    "attn_dispatch_resolve",
    "mha_decode_num_partitions",
    "mha_split_kv_decode_num_partitions",
    "mla_dispatch_args_scalar",
    "mla_dispatch_key",
    "multiprocessor_count",
]
