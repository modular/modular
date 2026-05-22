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

"""Unit tests for the BinaryNotification overflow warning helper.

`_warn_on_notif_overflow` is invoked after every successful
`post_transfer_request` in the libfabric path. It queries the agent for
the per-transfer overflow accounting populated by
`nixlLibfabricEngine::postXfer` and emits a structured `logger.warning`
when the BinaryNotification's fixed xfer_ids array could not hold every
RDMA xfer_id. Backends without a fixed-capacity notification (UCX)
report `dropped == 0` so the helper is a silent no-op.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    _warn_on_notif_overflow,
)


def _make_agent(dropped: int, submitted: int) -> MagicMock:
    agent = MagicMock()
    agent.get_transfer_notif_overflow.return_value = (dropped, submitted)
    return agent


def test_no_overflow_emits_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _make_agent(dropped=0, submitted=4096)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        _warn_on_notif_overflow(
            agent,
            transfer_id=123,
            transfer_name="abc-xfer",
            remote_agent="remote-0",
            src_replica_idx=0,
            dst_replica_idx=0,
            tp_idx=0,
            direction="write",
        )
    agent.get_transfer_notif_overflow.assert_called_once_with(123)
    assert caplog.records == []


def test_overflow_emits_structured_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _make_agent(dropped=7688, submitted=11784)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        _warn_on_notif_overflow(
            agent,
            transfer_id=42,
            transfer_name="xfer-name",
            remote_agent="remote-1",
            src_replica_idx=2,
            dst_replica_idx=5,
            tp_idx=3,
            direction="write",
        )
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    # The message must carry the ticket tag (used for log triage), both
    # counts, the remote agent, the transfer name, the src/dst DP
    # replica indices, and the TP shard so operators can pinpoint the
    # exact node pair on a DP x TP deployment without needing to
    # cross-reference other log lines.
    msg = record.getMessage()
    assert "[GEX-3736]" in msg
    assert "7688" in msg
    assert "11784" in msg
    assert "remote-1" in msg
    assert "xfer-name" in msg
    assert "src DP 2" in msg
    assert "dst DP 5" in msg
    assert "TP shard 3" in msg
    assert "write" in msg


def test_overflow_warning_distinguishes_read_direction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The read path (initiate_read_transfer) wires the helper with
    # direction="read"; verifying the formatted message echoes the
    # direction ensures ops can tell read-side breaches from write-side
    # breaches without inspecting the call site.
    agent = _make_agent(dropped=1, submitted=4097)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        _warn_on_notif_overflow(
            agent,
            transfer_id=7,
            transfer_name="read-xfer",
            remote_agent="remote-2",
            src_replica_idx=0,
            dst_replica_idx=0,
            tp_idx=0,
            direction="read",
        )
    assert len(caplog.records) == 1
    assert "read" in caplog.records[0].getMessage()
