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
"""Tests for EagleHiddenStateManager on multi-GPU."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import RequestID
from max.interfaces.context import SamplingParams
from max.interfaces.tokens import TokenBuffer
from max.pipelines.core.context import TextContext
from max.pipelines.lib.speculative_decoding.eagle_hidden_state_manager import (
    EagleHiddenStateManager,
)

HIDDEN_DIM = 64
MAX_LENGTH = 10_000
NUM_DRAFT_STEPS = 3
TOKENS_PER_REQUEST = NUM_DRAFT_STEPS + 1  # verification tokens per request


@pytest.fixture
def devices() -> list[Accelerator]:
    return [Accelerator(id=0), Accelerator(id=1)]


@pytest.fixture
def session(devices: list[Accelerator]) -> InferenceSession:
    return InferenceSession(devices=devices)


@pytest.fixture
def manager(
    devices: list[Accelerator], session: InferenceSession
) -> EagleHiddenStateManager:
    return EagleHiddenStateManager(
        hidden_dim=HIDDEN_DIM,
        dtype=DType.float32,
        devices=devices,
        max_batch_size=8,
        num_draft_steps=NUM_DRAFT_STEPS,
        session=session,
    )


def _make_context() -> TextContext:
    return TextContext(
        max_length=MAX_LENGTH,
        tokens=TokenBuffer(np.array([1, 2, 3], dtype=np.int64)),
        request_id=RequestID(),
        eos_token_ids=set(),
        sampling_params=SamplingParams(),
    )


def _to_device(arr: np.ndarray, device: Accelerator) -> Buffer:
    return Buffer.from_numpy(arr).to(device)


def _to_numpy(buf: Buffer) -> np.ndarray:
    return torch.from_dlpack(buf).cpu().numpy()


def test_save_extracted_and_get_draft_input(
    manager: EagleHiddenStateManager, devices: list[Accelerator]
) -> None:
    """save_extracted stores accepted rows; get_draft_input retrieves them."""
    ctxs = [_make_context() for _ in range(4)]
    splits = np.array([0, 2, 4], dtype=np.int64)
    first_rejected = np.array([1, 2, 0, 3], dtype=np.int64)
    # num_rows per request: [2, 3, 1, 4]

    logit_offsets = np.array(
        [i * TOKENS_PER_REQUEST for i in range(5)], dtype=np.int64
    )

    # Device 0: requests 0,1 → 2*TOKENS_PER_REQUEST rows
    dev0_rows = 2 * TOKENS_PER_REQUEST
    hs0 = np.zeros((dev0_rows, HIDDEN_DIM), dtype=np.float32)
    hs0[0] = 1.0  # Request 0, accepted row 0
    hs0[1] = 2.0  # Request 0, accepted row 1
    hs0[TOKENS_PER_REQUEST] = 3.0  # Request 1, accepted row 0
    hs0[TOKENS_PER_REQUEST + 1] = 4.0  # Request 1, accepted row 1
    hs0[TOKENS_PER_REQUEST + 2] = 5.0  # Request 1, accepted row 2

    # Device 1: requests 2,3 → 2*TOKENS_PER_REQUEST rows
    dev1_rows = 2 * TOKENS_PER_REQUEST
    hs1 = np.zeros((dev1_rows, HIDDEN_DIM), dtype=np.float32)
    hs1[0] = 6.0  # Request 2, accepted row 0
    hs1[TOKENS_PER_REQUEST] = 7.0  # Request 3, accepted row 0
    hs1[TOKENS_PER_REQUEST + 1] = 8.0  # Request 3, accepted row 1
    hs1[TOKENS_PER_REQUEST + 2] = 9.0  # Request 3, accepted row 2
    hs1[TOKENS_PER_REQUEST + 3] = 10.0  # Request 3, accepted row 3

    target_hs = [_to_device(hs0, devices[0]), _to_device(hs1, devices[1])]
    manager.save_extracted(
        ctxs, target_hs, logit_offsets, first_rejected, splits
    )

    replica_batches = [ctxs[:2], ctxs[2:]]
    result = manager.get_draft_input(replica_batches)

    # Device 0: request 0 (2 rows) + request 1 (3 rows) = 5 rows
    r0 = _to_numpy(result[0])
    assert r0.shape == (5, HIDDEN_DIM)
    np.testing.assert_array_equal(r0[0], 1.0)
    np.testing.assert_array_equal(r0[1], 2.0)
    np.testing.assert_array_equal(r0[2], 3.0)
    np.testing.assert_array_equal(r0[3], 4.0)
    np.testing.assert_array_equal(r0[4], 5.0)

    # Device 1: request 2 (1 row) + request 3 (4 rows) = 5 rows
    r1 = _to_numpy(result[1])
    assert r1.shape == (5, HIDDEN_DIM)
    np.testing.assert_array_equal(r1[0], 6.0)
    np.testing.assert_array_equal(r1[1], 7.0)
    np.testing.assert_array_equal(r1[2], 8.0)
    np.testing.assert_array_equal(r1[3], 9.0)
    np.testing.assert_array_equal(r1[4], 10.0)


def test_gather_on_batch_change(
    manager: EagleHiddenStateManager, devices: list[Accelerator]
) -> None:
    """When a request is removed, get_draft_input gathers surviving rows."""
    ctxs = [_make_context() for _ in range(4)]
    splits = np.array([0, 2, 4], dtype=np.int64)
    first_rejected = np.array([1, 2, 0, 3], dtype=np.int64)
    # num_rows per request: [2, 3, 1, 4]

    logit_offsets = np.array(
        [i * TOKENS_PER_REQUEST for i in range(5)], dtype=np.int64
    )

    dev0_rows = 2 * TOKENS_PER_REQUEST
    hs0 = np.zeros((dev0_rows, HIDDEN_DIM), dtype=np.float32)
    hs0[0:2] = 1.0  # Request 0 accepted rows
    hs0[TOKENS_PER_REQUEST : TOKENS_PER_REQUEST + 3] = 2.0  # Request 1

    dev1_rows = 2 * TOKENS_PER_REQUEST
    hs1 = np.zeros((dev1_rows, HIDDEN_DIM), dtype=np.float32)
    hs1[0:1] = 3.0  # Request 2
    hs1[TOKENS_PER_REQUEST : TOKENS_PER_REQUEST + 4] = 4.0  # Request 3

    target_hs = [_to_device(hs0, devices[0]), _to_device(hs1, devices[1])]
    manager.save_extracted(
        ctxs, target_hs, logit_offsets, first_rejected, splits
    )

    # Request 1 (on device 0) terminates.
    manager.release(ctxs[1].request_id)

    # Surviving: [ctx0] on device 0, [ctx2, ctx3] on device 1.
    replica_batches = [[ctxs[0]], [ctxs[2], ctxs[3]]]
    result = manager.get_draft_input(replica_batches)

    # Device 0: only request 0's rows
    r0 = _to_numpy(result[0])
    assert r0.shape == (2, HIDDEN_DIM)
    np.testing.assert_array_equal(r0, 1.0)

    # Device 1: request 2 + request 3 (unchanged)
    r1 = _to_numpy(result[1])
    assert r1.shape == (5, HIDDEN_DIM)
    np.testing.assert_array_equal(r1[0:1], 3.0)
    np.testing.assert_array_equal(r1[1:5], 4.0)


def test_cross_device_raises(
    manager: EagleHiddenStateManager, devices: list[Accelerator]
) -> None:
    """Cross-device migration is not supported and raises ValueError."""
    ctxs = [_make_context() for _ in range(4)]
    splits = np.array([0, 2, 4], dtype=np.int64)
    first_rejected = np.array([0, 0, 0, 0], dtype=np.int64)
    # num_rows per request: [1, 1, 1, 1]

    logit_offsets = np.array(
        [i * TOKENS_PER_REQUEST for i in range(5)], dtype=np.int64
    )

    hs0 = np.full((2 * TOKENS_PER_REQUEST, HIDDEN_DIM), 10.0, dtype=np.float32)
    hs1 = np.full((2 * TOKENS_PER_REQUEST, HIDDEN_DIM), 20.0, dtype=np.float32)
    target_hs = [_to_device(hs0, devices[0]), _to_device(hs1, devices[1])]
    manager.save_extracted(
        ctxs, target_hs, logit_offsets, first_rejected, splits
    )

    # Swap devices: should raise since requests can't migrate.
    replica_batches = [[ctxs[2], ctxs[3]], [ctxs[0], ctxs[1]]]
    with pytest.raises(ValueError, match="Cross-device migration"):
        manager.get_draft_input(replica_batches)
