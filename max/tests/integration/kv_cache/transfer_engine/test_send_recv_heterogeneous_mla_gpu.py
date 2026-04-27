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

"""Heterogeneous MLA KV transfer: sender (DP=1,TP=2) <-> receiver (DP=2,TP=1).

The sender declares ``replicate_kv_across_tp=True`` (MLA, num_kv_heads=1)
with TP=2, while the receiver is DP=2/TP=1. At connect() time the engines
agree on an effective DP=2/TP=1 view via ``resolve_peer_view`` setting
``flatten_local=True`` on the sender, so each of the sender's TP shards
is treated as a standalone DP replica. ``initiate_send_transfer`` picks
sender shard 0.

The test validates:
  - Heterogeneous shape negotiation at connect().
  - End-to-end data transfer from sender's single replica to each of the
    receiver's 2 replicas.
  - The send uses exactly one sender shard (transfer_ids length == 1).

Uses 4 GPUs total (2 for sender, 2 for receiver).
"""

import multiprocessing as mp

import numpy as np
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.kv_cache import KVTransferEngine


def paged(
    total_bytes: int,
    page_values: list[int],
    accelerator_idx: int,
) -> Buffer:
    """Create a buffer with distinct values per page."""
    total_num_pages = len(page_values)
    page_size = total_bytes // total_num_pages
    # uint8 keeps the buffer one byte per element while still fitting
    # sentinel values up to 255.
    arr = np.empty(total_bytes, dtype=np.uint8)
    for i, v in enumerate(page_values):
        arr[i * page_size : (i + 1) * page_size] = v
    return Buffer.from_numpy(arr).to(Accelerator(accelerator_idx))


def sender_routine(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queues: list[mp.Queue],  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Sender: DP=1, TP=2, MLA (replicate_kv_across_tp=True).

    KV is replicated across both TP shards; the engine will pick shard 0
    for actual transfers under flatten_local.
    """
    # KV is logically identical on all shards (MLA); we use distinct
    # page values anyway so the test can detect "wrong shard picked."
    replica_0_tensors = [
        paged(total_bytes, page_values=[100, 101], accelerator_idx=0),
        paged(total_bytes, page_values=[200, 201], accelerator_idx=1),
    ]

    engine = KVTransferEngine(
        "sender_engine",
        [replica_0_tensors],
        total_num_pages=total_num_pages,
        replicate_kv_across_tp=True,
    )

    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # One send per receiver replica (2 sends total).
    for dst_replica_idx in range(2):
        transfer_req = engine.initiate_send_transfer(
            remote_md,
            src_idxs=[0, 1],
            dst_idxs=[0, 1],
            src_replica_idx=0,
            dst_replica_idx=dst_replica_idx,
        )
        # Under flatten_local we should post exactly ONE NIXL request
        # (not one per TP shard — that would duplicate replicated KV).
        assert len(transfer_req.transfer_ids) == 1, (
            f"Expected 1 transfer_id under flatten_local, got "
            f"{len(transfer_req.transfer_ids)}"
        )
        assert transfer_req.tp_shard_count == 1
        transfer_queues[dst_replica_idx].put(transfer_req)
        engine.sync_and_release(transfer_req)

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine.cleanup()


def receiver_routine(
    sender_md_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue,  # type: ignore[type-arg]
    transfer_queues: list[mp.Queue],  # type: ignore[type-arg]
    sender_done_queue: mp.Queue,  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue,  # type: ignore[type-arg]
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Receiver: DP=2, TP=1, MLA (replicate_kv_across_tp=True).

    Each of the 2 replicas has a single tensor. Receiver is not the
    flattening side (TP=1), but both sides advertise MLA so connect
    still traverses the resolve_peer_view MLA-heterogeneous branch.
    """
    replicas = [
        [paged(total_bytes, page_values=[99, 99], accelerator_idx=2 + r)]
        for r in range(2)
    ]

    engine = KVTransferEngine(
        "receiver_engine",
        replicas,
        total_num_pages=total_num_pages,
        replicate_kv_across_tp=True,
    )

    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    for dst_replica_idx in range(2):
        transfer_req = transfer_queues[dst_replica_idx].get()
        engine.sync_and_release(transfer_req)

    # Sender always picks local shard 0 -> replica_0_tensors[0] (values
    # [100, 101]). Every receiver replica should now hold those values.
    page_size = total_bytes // total_num_pages
    for replica_idx, replica_tensors in enumerate(replicas):
        shard = replica_tensors[0]
        result = shard.to_numpy()
        for page_idx, expected in enumerate([100, 101]):
            actual = result[page_idx * page_size : (page_idx + 1) * page_size]
            assert (actual == expected).all(), (
                f"Receiver replica {replica_idx} page {page_idx}: "
                f"expected {expected}, got {actual[0]}"
            )

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_heterogeneous_mla_dp1tp2_to_dp2tp1() -> None:
    """Heterogeneous MLA DI: prefill (DP=1,TP=2) <-> decode (DP=2,TP=1).

    Uses 4 GPUs (2 per process). Validates the full connect/send/recv
    loop under flatten_local on the sender side.
    """
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_md_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    transfer_queues: list[mp.Queue] = [ctx.Queue() for _ in range(2)]  # type: ignore[type-arg]
    sender_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    receiver_done_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]

    GB = 1024 * 1024 * 1024
    total_bytes = int(0.5 * GB)
    total_num_pages = 2

    sender_proc = ctx.Process(
        target=sender_routine,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queues,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            total_bytes,
        ),
    )
    receiver_proc = ctx.Process(
        target=receiver_routine,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queues,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            total_bytes,
        ),
    )

    sender_proc.start()
    receiver_proc.start()

    sender_proc.join()
    receiver_proc.join()

    assert sender_proc.exitcode == 0, (
        f"Sender process failed with exit code {sender_proc.exitcode}"
    )
    assert receiver_proc.exitcode == 0, (
        f"Receiver process failed with exit code {receiver_proc.exitcode}"
    )
