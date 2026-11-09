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
"""Paged caches and tiny configs for the K3 DSpark drafter's kernel tests.

Generalizes ``kimi_k3_modulev3``'s ``_paged_cache_test_utils`` to several
devices: one real, single-device pool per device, wrapped into the mesh view
the drafter's kernels take. A draft step reads the same pool at two lengths
(the context store's and the block forward's), so :func:`allocate_dspark_cache`
returns a view factory over one shared allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from _dspark_test_data import TINY_DRAFT_CONFIG
from max.driver import Buffer, Device
from max.dtype import DType
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    PACKED_PAGE_STRIDE,
    KVCacheParams,
    MLAAttnKey,
    padded_lut_cols,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.config import (
    K3DSparkDraftConfig,
    parse_k3_dspark_config,
)
from max.pipelines.architectures.kimi_k3_modulev3.dspark.dspark import (
    K3DSpark,
    K3DSparkLayer,
)
from max.support.math import ceildiv
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import DeviceMapping, DeviceMesh


def _zeros(shape: tuple[int, ...], dtype: DType, device: Device) -> Tensor:
    """A real, zero-filled Tensor of any rank, as ``Buffer.view`` needs."""
    raw = np.zeros(
        (*shape[:-1], shape[-1] * dtype.size_in_bytes), dtype=np.uint8
    )
    buf = Buffer.from_dlpack(raw).to(device).view(dtype=dtype, shape=shape)
    return Tensor(storage=buf)


def allocate_dspark_cache(
    params: KVCacheParams, total_rows: Sequence[int]
) -> Callable[[Sequence[int], Sequence[int]], PagedCacheValues]:
    """Allocates one real paged pool per device and returns its view factory.

    Args:
        params: The drafter's leaf, whose devices each get their own pool.
        total_rows: The widest request lengths any view will hold; the pages
            and lookup table are sized for these.

    Returns:
        A callable ``(prompt_lens, cache_lengths)`` building the mesh view
        of the shared pools at those lengths.
    """
    page_size = params.page_size
    blocks: list[list[int]] = []
    next_block = 0
    for seq_len in total_rows:
        num_blocks = ceildiv(seq_len, page_size)
        blocks.append(list(range(next_block, next_block + num_blocks)))
        next_block += num_blocks
    total_num_pages = next_block
    # One page past the pool backs the null block the lookup table's tail
    # padding points at; the fill value must be exactly that page's id, or
    # the kernel's table read strides out of bounds.
    num_cols = padded_lut_cols(ceildiv(max(total_rows), page_size))
    lookup_np = np.full((len(blocks), num_cols), total_num_pages, np.uint32)
    for row, req_blocks in enumerate(blocks):
        lookup_np[row, : len(req_blocks)] = req_blocks
    devices = [d.to_device() for d in params.devices]
    mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
    mapping = DeviceMapping.replicated(mesh)
    with ensure_context():
        pools = [
            _zeros(
                (total_num_pages + 1, *params.shape_per_block),
                params.dtype,
                device,
            )
            for device in devices
        ]

    def view(
        prompt_lens: Sequence[int], cache_lengths: Sequence[int]
    ) -> PagedCacheValues:
        seq_lens = [
            cache_len + prompt_len
            for cache_len, prompt_len in zip(
                cache_lengths, prompt_lens, strict=True
            )
        ]
        max_prompt_length = max(prompt_lens)
        max_cache_length = max(seq_lens)
        with ensure_context():
            attn_key = params.resolve_attn_key(
                len(prompt_lens), max_prompt_length, max_cache_length
            )
            per_device = []
            for pool, device in zip(pools, devices, strict=True):
                per_device.append(
                    PagedCacheValues(
                        kv_blocks=pool,
                        cache_lengths=Tensor(
                            storage=Buffer.from_numpy(
                                np.array(cache_lengths, np.uint32)
                            ).to(device)
                        ),
                        lookup_table=Tensor(
                            storage=Buffer.from_numpy(lookup_np).to(device)
                        ),
                        # The kernels read page stride, prompt/cache
                        # length and the MLA partition count by value on
                        # the host, and the cache's declared symbolic
                        # inputs type them on the CPU -- staging them on
                        # the GPU leaves the kernel dereferencing a device
                        # mapping host-side, which AMD's large-BAR VRAM
                        # permits and CUDA faults on (SIGSEGV,
                        # SEGV_ACCERR).
                        max_prompt_length=Tensor(
                            storage=Buffer.from_numpy(
                                np.array([max_prompt_length], np.uint32)
                            )
                        ),
                        max_cache_length=Tensor(
                            storage=Buffer.from_numpy(
                                np.array([max_cache_length], np.uint32)
                            )
                        ),
                        page_stride=Tensor(
                            storage=Buffer.from_numpy(
                                np.array([PACKED_PAGE_STRIDE], np.int64)
                            )
                        ),
                        attention_dispatch_metadata=Tensor(
                            storage=attn_key.pack_into_buffer(
                                device, max_cache_length
                            )
                        ),
                        mla_num_partitions=(
                            Tensor(
                                storage=Buffer.from_numpy(
                                    np.array(
                                        [attn_key.num_partitions], np.int64
                                    )
                                )
                            )
                            if isinstance(attn_key, MLAAttnKey)
                            else None
                        ),
                    )
                )
            return PagedCacheValues.from_upstream(per_device, mapping)

    return view


def tiny_dspark_config(*, num_layers: int = 2) -> K3DSparkDraftConfig:
    """A tiny ``k3_dspark`` config sized for two-device kernel tests."""
    config = dict(TINY_DRAFT_CONFIG)
    config["num_hidden_layers"] = num_layers
    return parse_k3_dspark_config(config)


def set_dspark_weight(draft: K3DSpark, name: str, value: Tensor) -> None:
    """Writes ``value`` to the weight ``name`` names, module-tree path."""
    parts = name.split(".")
    parent: K3DSpark | K3DSparkLayer = draft
    if parts[0] == "layers":
        parent = draft.layers[int(parts[1])]
        parts = parts[2:]
    for part in parts[:-1]:
        parent = getattr(parent, part)
    weight = getattr(parent, parts[-1])
    setattr(parent, parts[-1], value.cast(weight.dtype))
