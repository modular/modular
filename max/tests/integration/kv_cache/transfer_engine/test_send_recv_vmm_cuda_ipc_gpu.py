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

"""Same-node transfers of VMM-backed buffers go over CUDA IPC.

The BUILD target forces the VMM memory manager with 2 MiB granules, so each
buffer spans many CUDA allocations, and keeps UCX off InfiniBand. If the
granules are not exportable, UCX stages GPU data through host memory, which the
bandwidth floor catches; if the engine registers a range that spans
allocations, the transfer crashes or corrupts pages, which the content check
catches.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
from _transfer_engine_helpers import every_leaf, kv_memory
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import KVTransferEngine

# 1.5 MiB pages do not divide the 2 MiB granules, so some pages straddle a
# granule boundary on each side, at different offsets.
BYTES_PER_PAGE = 3 * 512 * 1024
TOTAL_NUM_PAGES = 256
SRC_IDXS = list(range(TOTAL_NUM_PAGES))
DST_IDXS = list(reversed(SRC_IDXS))

# Well under NVLink cuda_ipc and well over host staging through tcp, so only
# the cuda_ipc path clears it.
MIN_BANDWIDTH_GIB_S = 10.0


def _page_pattern() -> np.ndarray:
    pages = np.arange(TOTAL_NUM_PAGES, dtype=np.int64) % 127 + 1
    return np.repeat(pages.astype(np.int8), BYTES_PER_PAGE)


def _sender(
    md_out: mp.Queue,  # type: ignore[type-arg]
    md_in: mp.Queue,  # type: ignore[type-arg]
    req_out: mp.Queue,  # type: ignore[type-arg]
    bandwidth_out: mp.Queue,  # type: ignore[type-arg]
    done: mp.Queue,  # type: ignore[type-arg]
) -> None:
    buf = Buffer.from_numpy(_page_pattern()).to(Accelerator(0))
    engine = KVTransferEngine("vmm_sender", [[kv_memory(buf, TOTAL_NUM_PAGES)]])
    assert engine.tensor_agents[0][0].region_splits[0], (
        "expected the VMM-backed buffer to span several CUDA allocations"
    )
    md_out.put(engine.metadata)
    remote = md_in.get()
    engine.connect(remote)

    # The first transfer maps the peer's allocations; time the second.
    for timed in (False, True):
        start = time.perf_counter()
        req = engine.initiate_send_transfer(
            remote,
            every_leaf(remote, SRC_IDXS),
            every_leaf(remote, DST_IDXS),
            src_replica_idx=0,
            dst_replica_idx=0,
        )
        req_out.put(req)
        engine.sync_and_release(req)
        if timed:
            elapsed = time.perf_counter() - start
            bandwidth_out.put(
                TOTAL_NUM_PAGES * BYTES_PER_PAGE / elapsed / 2**30
            )
    done.get()
    engine.cleanup()


def _receiver(
    md_out: mp.Queue,  # type: ignore[type-arg]
    md_in: mp.Queue,  # type: ignore[type-arg]
    req_in: mp.Queue,  # type: ignore[type-arg]
    done: mp.Queue,  # type: ignore[type-arg]
) -> None:
    buf = Buffer.from_numpy(
        np.zeros(TOTAL_NUM_PAGES * BYTES_PER_PAGE, dtype=np.int8)
    ).to(Accelerator(1))
    engine = KVTransferEngine(
        "vmm_receiver", [[kv_memory(buf, TOTAL_NUM_PAGES)]]
    )
    md_out.put(engine.metadata)
    remote = md_in.get()
    engine.connect(remote)
    for _ in range(2):
        engine.sync_and_release(req_in.get())

    received = buf.to_numpy().reshape(TOTAL_NUM_PAGES, BYTES_PER_PAGE)
    sent = _page_pattern().reshape(TOTAL_NUM_PAGES, BYTES_PER_PAGE)
    for src_idx, dst_idx in zip(SRC_IDXS, DST_IDXS, strict=True):
        assert np.array_equal(received[dst_idx], sent[src_idx]), (
            f"page {src_idx} -> {dst_idx} arrived corrupted"
        )
    done.put(None)
    engine.cleanup()


def test_vmm_buffers_transfer_over_cuda_ipc() -> None:
    ctx = mp.get_context("spawn")
    sender_md: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    requests: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    bandwidth: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    done: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    sender = ctx.Process(
        target=_sender,
        args=(sender_md, receiver_md, requests, bandwidth, done),
    )
    receiver = ctx.Process(
        target=_receiver, args=(receiver_md, sender_md, requests, done)
    )
    sender.start()
    receiver.start()
    sender.join(timeout=300)
    receiver.join(timeout=300)

    assert sender.exitcode == 0, f"sender exited with {sender.exitcode}"
    assert receiver.exitcode == 0, f"receiver exited with {receiver.exitcode}"
    gib_s = bandwidth.get(timeout=1)
    print(f"VMM same-node transfer: {gib_s:.1f} GiB/s")
    assert gib_s >= MIN_BANDWIDTH_GIB_S, (
        f"{gib_s:.1f} GiB/s is below {MIN_BANDWIDTH_GIB_S} GiB/s; GPU data "
        "is probably being staged through host memory instead of cuda_ipc"
    )
