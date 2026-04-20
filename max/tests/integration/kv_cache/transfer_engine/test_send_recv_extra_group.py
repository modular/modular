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

# NIXL transfers can fail when multiple transfers are in the same bazel pytest
# target.  This transfer test is alone in this file.

from queue import Queue
from threading import Thread

import numpy as np
import pytest
from max.driver import CPU, Device
from max.driver.buffer import Buffer
from max.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)


def transfer_routine_sender(
    engine: KVTransferEngine,
    remote: KVTransferEngineMetadata,
    queue: Queue,  # type: ignore[type-arg]
    src_idxs: list[int],
    dst_idxs: list[int],
    src_replica_idx: int,
    dst_replica_idx: int,
) -> None:
    transfer_req = engine.initiate_send_transfer(
        remote, src_idxs, dst_idxs, src_replica_idx, dst_replica_idx
    )
    queue.put(transfer_req)
    engine.sync_and_release(transfer_req)


def transfer_routine_receiver(engine: KVTransferEngine, queue: Queue) -> None:  # type: ignore[type-arg]
    transfer_req = queue.get()
    engine.sync_and_release(transfer_req)


@pytest.mark.parametrize("device", [CPU()])
def test_send_recv_with_extra_group(device: Device) -> None:
    """Transfer with an extra tensor group moves both primary and group data.

    Registers a "draft" extra group on both engines, then verifies that a
    single initiate_send_transfer copies pages from both the primary tensor
    and the draft tensor atomically.
    """
    total_num_pages = 3
    primary_elts_per_page = 4
    draft_elts_per_page = 2
    primary_num_elts = total_num_pages * primary_elts_per_page
    draft_num_elts = total_num_pages * draft_elts_per_page

    # Primary (target) KV buffers
    primary_1 = Buffer.from_numpy(
        np.arange(primary_num_elts, dtype=np.int16) + 10
    ).to(device)
    primary_2 = Buffer.from_numpy(
        np.arange(primary_num_elts, dtype=np.int16) + 80
    ).to(device)

    # Draft KV buffers (smaller — fewer layers)
    draft_1 = Buffer.from_numpy(
        np.arange(draft_num_elts, dtype=np.int16) + 200
    ).to(device)
    draft_2 = Buffer.from_numpy(
        np.arange(draft_num_elts, dtype=np.int16) + 300
    ).to(device)

    # Create engines with primary tensors (DP=1, TP=1)
    engine_1 = KVTransferEngine(
        "engine_1",
        [[primary_1]],
        total_num_pages=total_num_pages,
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [[primary_2]],
        total_num_pages=total_num_pages,
    )

    # Register draft group on both engines
    engine_1.register_tensor_group(
        "draft", [[draft_1]], total_num_pages=total_num_pages
    )
    engine_2.register_tensor_group(
        "draft", [[draft_2]], total_num_pages=total_num_pages
    )

    # Connect after registration so metadata includes extra groups
    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    # Transfer page 2 from engine_1 to page 0 of engine_2
    queue: Queue[TransferReqData] = Queue()
    src_idxs = [2]
    dst_idxs = [0]

    thread_1 = Thread(
        target=transfer_routine_sender,
        args=(engine_1, engine_2.metadata, queue, src_idxs, dst_idxs, 0, 0),
    )
    thread_2 = Thread(target=transfer_routine_receiver, args=(engine_2, queue))

    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    # Verify primary: page 2 of engine_1 → page 0 of engine_2
    # engine_1 primary page 2: [18, 19, 20, 21]
    expected_primary_2 = np.array(
        [18, 19, 20, 21, 84, 85, 86, 87, 88, 89, 90, 91],
        dtype=np.int16,
    )
    assert np.array_equal(primary_2.to_numpy(), expected_primary_2)

    # Verify draft: page 2 of engine_1 → page 0 of engine_2
    # engine_1 draft page 2: [204, 205]
    expected_draft_2 = np.array(
        [204, 205, 302, 303, 304, 305],
        dtype=np.int16,
    )
    assert np.array_equal(draft_2.to_numpy(), expected_draft_2)

    # Source buffers should be unchanged
    expected_primary_1 = np.arange(primary_num_elts, dtype=np.int16) + 10
    expected_draft_1 = np.arange(draft_num_elts, dtype=np.int16) + 200
    assert np.array_equal(primary_1.to_numpy(), expected_primary_1)
    assert np.array_equal(draft_1.to_numpy(), expected_draft_1)

    engine_2.cleanup()
    engine_1.cleanup()
