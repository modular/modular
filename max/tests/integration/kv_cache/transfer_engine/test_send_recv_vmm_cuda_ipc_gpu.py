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
granules are not exportable, UCX moves GPU data through host memory over tcp,
which the protocol check catches; if the engine registers a range that spans
allocations, the transfer crashes or corrupts pages, which the content check
catches.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
from _transfer_engine_helpers import every_leaf, join_peers, kv_memory
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.pipelines.kv_cache import KVTransferEngine

# 1.5 MiB pages do not divide the 2 MiB granules, so some pages straddle a
# granule boundary on each side, at different offsets.
BYTES_PER_PAGE = 3 * 512 * 1024
TOTAL_NUM_PAGES = 256
SRC_IDXS = list(range(TOTAL_NUM_PAGES))
DST_IDXS = list(reversed(SRC_IDXS))

_GPU_PUT_HEADER = re.compile(r"ucp_put\S* from cuda/GPU\d+ to cuda\b")


def _page_pattern() -> np.ndarray:
    pages = np.arange(TOTAL_NUM_PAGES, dtype=np.int64) % 127 + 1
    return np.repeat(pages.astype(np.int8), BYTES_PER_PAGE)


def _gpu_put_transports(ucx_log: str) -> list[str]:
    """Returns the transport of every row in UCX's GPU-to-GPU put tables.

    With ``UCX_PROTO_INFO=y``, UCX logs one table per endpoint and operation
    when it first selects a protocol. A row ends with the transport, such as
    ``cuda_ipc/cuda`` for NVLink or ``tcp/lo`` for host staging.
    """
    transports = []
    lines = ucx_log.splitlines()
    for i, line in enumerate(lines):
        if not _GPU_PUT_HEADER.search(line):
            continue
        # Skip the header's closing border, then read rows to the next border.
        for row in lines[i + 2 :]:
            cells = [c.strip() for c in row.split("|")[1:-1]]
            if len(cells) < 3:
                break
            transports.append(cells[-1])
    return transports


def _sender(
    md_out: mp.Queue,  # type: ignore[type-arg]
    md_in: mp.Queue,  # type: ignore[type-arg]
    req_out: mp.Queue,  # type: ignore[type-arg]
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

    req = engine.initiate_send_transfer(
        remote,
        every_leaf(remote, SRC_IDXS),
        every_leaf(remote, DST_IDXS),
        src_replica_idx=0,
        dst_replica_idx=0,
    )
    req_out.put(req)
    engine.sync_and_release(req)
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
    engine.sync_and_release(req_in.get())

    received = buf.to_numpy().reshape(TOTAL_NUM_PAGES, BYTES_PER_PAGE)
    sent = _page_pattern().reshape(TOTAL_NUM_PAGES, BYTES_PER_PAGE)
    for src_idx, dst_idx in zip(SRC_IDXS, DST_IDXS, strict=True):
        assert np.array_equal(received[dst_idx], sent[src_idx]), (
            f"page {src_idx} -> {dst_idx} arrived corrupted"
        )
    done.put(None)
    engine.cleanup()


def test_vmm_buffers_transfer_over_cuda_ipc(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    sender_md: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    requests: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    done: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    ucx_log_path = tmp_path / "sender_ucx.log"

    sender = ctx.Process(
        target=_sender,
        args=(sender_md, receiver_md, requests, done),
    )
    receiver = ctx.Process(
        target=_receiver, args=(receiver_md, sender_md, requests, done)
    )
    # A spawned child copies the env at start(), so only the sender logs.
    os.environ.update(UCX_PROTO_INFO="y", UCX_LOG_FILE=str(ucx_log_path))
    sender.start()
    del os.environ["UCX_PROTO_INFO"], os.environ["UCX_LOG_FILE"]
    receiver.start()
    join_peers([sender, receiver])
    transports = _gpu_put_transports(ucx_log_path.read_text())
    # UCX is pinned by @ucx_prebuilt, so the table format changes only in a
    # UCX upgrade, which reruns this test.
    assert transports, (
        f"found no GPU-to-GPU put table in {ucx_log_path}; if UCX was "
        "upgraded, update _GPU_PUT_HEADER and _gpu_put_transports to its "
        "UCX_PROTO_INFO format"
    )
    assert all(t.startswith("cuda_ipc/") for t in transports), (
        f"GPU-to-GPU puts selected {transports}, not cuda_ipc; GPU data is "
        "being staged through host memory"
    )
