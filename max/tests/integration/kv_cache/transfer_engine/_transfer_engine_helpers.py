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

import time
from collections.abc import Sequence
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.pipelines.kv_cache import KVTransferEngineMetadata

# Under bazel's default 300s medium-test timeout, so a stuck peer fails the
# test with its captured output rather than being killed as a TIMEOUT.
_JOIN_TIMEOUT_S = 240.0


def view_2d_uint8(buf: Buffer, total_num_pages: int) -> Buffer:
    """Views a raw buffer as a 2-D uint8 ``[total_num_pages, bytes_per_page]`` array."""
    bytes_per_page = (
        buf.num_elements * buf.dtype.size_in_bytes // total_num_pages
    )
    return buf.view(DType.uint8, [total_num_pages, bytes_per_page])


def kv_memory(buf: Buffer, total_num_pages: int) -> KVCacheMemory:
    """Wraps a single raw buffer as a one-shard, non-replicated NIXL group."""
    return kv_group([buf], total_num_pages)


def kv_group(
    bufs: Sequence[Buffer],
    total_num_pages: int,
    *,
    replicated: bool = False,
) -> KVCacheMemory:
    """Wraps raw TP-shard buffers as one authored NIXL group.

    Every buffer must share a shape after the uint8 page-view; the group carries
    all shards of one logical ``(child, kind)`` tensor.
    """
    return KVCacheMemory(
        replicated=replicated,
        buffers=[view_2d_uint8(b, total_num_pages) for b in bufs],
    )


def every_leaf(
    remote: KVTransferEngineMetadata, idxs: Sequence[int]
) -> dict[str, list[int]]:
    """Addresses every group of ``remote`` with the same page indices."""
    return {leaf_id: list(idxs) for leaf_id in remote.leaf_ids}


def join_peers(
    procs: Sequence[BaseProcess], timeout_s: float = _JOIN_TIMEOUT_S
) -> None:
    """Joins ``procs``, killing the rest once one fails or ``timeout_s`` passes.

    The peers block on each other through queues, so one failed process would
    otherwise leave the others waiting forever.
    """
    deadline = time.monotonic() + timeout_s
    while not any(p.exitcode for p in procs):
        running = [p for p in procs if p.exitcode is None]
        remaining = deadline - time.monotonic()
        if not running or remaining <= 0:
            break
        wait([p.sentinel for p in running], timeout=remaining)
    killed = [p for p in procs if p.exitcode is None]
    for p in killed:
        p.kill()
        p.join()
    exit_codes = {p.name: p.exitcode for p in procs}
    assert all(code == 0 for code in exit_codes.values()), (
        f"Transfer processes exited with {exit_codes}; killed "
        f"{[p.name for p in killed]} after a peer failed or "
        f"{timeout_s:.0f}s elapsed"
    )
