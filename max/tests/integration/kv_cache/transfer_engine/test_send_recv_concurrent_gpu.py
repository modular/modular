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

import multiprocessing as mp
import time

import numpy as np
from max.driver import Accelerator
from max.driver.buffer import Buffer
from max.kv_cache import KVTransferEngine, TransferReqData

"""
This test launches 32 concurrent transfers at once.

Note: The payload size is intentionally kept small (512MB) to ensure the test
completes in a reasonable time while still validating concurrent transfer logic.
"""


def transfer_routine_sender(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    transfer_queue: mp.Queue,
    sender_done_queue: mp.Queue,
    receiver_done_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
    MB: int,
) -> None:
    device = Accelerator(0)

    blocks_np = np.full(total_bytes, 42, dtype=np.int8)
    blocks = Buffer.from_numpy(blocks_np).to(device)

    # Create engine (DP=1, TP=1)
    engine = KVTransferEngine(
        "engine_1", [[blocks]], total_num_pages=total_num_pages
    )

    # Connect with peer
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    t0 = time.time()
    transfer_reqs: list[TransferReqData] = []

    for idx in range(total_num_pages):
        transfer_req = engine.initiate_send_transfer(
            remote_md, [idx], [idx], src_replica_idx=0, dst_replica_idx=0
        )
        transfer_queue.put(transfer_req)
        transfer_reqs.append(transfer_req)

    for transfer_req in transfer_reqs:
        engine.sync_and_release(transfer_req)

    t1 = time.time()
    bw = total_bytes / (t1 - t0) / MB
    ms = (t1 - t0) * 1000

    print(
        f"[SENDER] Transferring {total_bytes / MB:.2f} MB took {ms:.2f} ms ({bw:.2f} MB/s)"
    )

    # Verify results
    assert (blocks.to_numpy() == 42).all()

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    transfer_queue: mp.Queue,
    sender_done_queue: mp.Queue,
    receiver_done_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
) -> None:
    device = Accelerator(1)

    blocks_np = np.full(total_bytes, 99, dtype=np.int8)
    blocks = Buffer.from_numpy(blocks_np).to(device)

    # Create engine (DP=1, TP=1)
    engine = KVTransferEngine(
        "engine_2", [[blocks]], total_num_pages=total_num_pages
    )

    # Connect with peer
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    for _ in range(total_num_pages):
        transfer_req = transfer_queue.get()
        engine.sync_and_release(transfer_req)

    # TODO: Verify results
    # assert (blocks.to_numpy() == 42).all()

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_send_recv_basic() -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()
    receiver_md_queue: mp.Queue = ctx.Queue()
    transfer_queue: mp.Queue = ctx.Queue()
    sender_done_queue: mp.Queue = ctx.Queue()
    receiver_done_queue: mp.Queue = ctx.Queue()

    # Transfer parameters
    MB = 1024 * 1024
    total_bytes = int(512 * MB)  # 512MB - reduced from 6GB for faster CI runs
    total_num_pages = 32

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue,
            sender_done_queue,
            receiver_done_queue,
            total_num_pages,
            total_bytes,
            MB,
        ),
    )
    receiver_proc = ctx.Process(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue,
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
