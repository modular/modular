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

"""Unit tests for ``DecodeScheduler._evict_expired_requests``.

Tests are written against the unbound method with a mocked ``self`` so
we avoid the full PD test harness for what is otherwise a small,
self-contained sweep.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from max.interfaces import SchedulerResult
from max.serve.scheduler.decode_scheduler import (
    DecodeScheduler,
    PendingPrefill,
    PendingTransfer,
)


def _make_self(
    *,
    ttl_s: float | None,
    prefill_reqs: dict[str, PendingPrefill],
    inflight_transfers: dict[str, PendingTransfer],
    prefill_reqs_per_replica: list[int],
) -> SimpleNamespace:
    self_obj = SimpleNamespace(
        scheduler_config=SimpleNamespace(decode_request_ttl_s=ttl_s),
        prefill_reqs=prefill_reqs,
        inflight_transfers=inflight_transfers,
        prefill_reqs_per_replica=prefill_reqs_per_replica,
        kv_cache=MagicMock(),
        transfer_engine=MagicMock(),
        response_queue=MagicMock(),
        dispatcher=MagicMock(),
    )
    # Bind the real instance method so the cancel-to-prefill path runs.
    self_obj._send_cancel_to_prefill = (
        DecodeScheduler._send_cancel_to_prefill.__get__(self_obj)
    )
    return self_obj


def _ctx(
    req_id: str, target_endpoint: str = "tcp://prefill:6000"
) -> SimpleNamespace:
    return SimpleNamespace(request_id=req_id, target_endpoint=target_endpoint)


def _pending(req_id: str, replica_idx: int, sent_at: float) -> PendingPrefill:
    return PendingPrefill(
        context=_ctx(req_id),  # type: ignore[arg-type]
        replica_idx=replica_idx,
        sent_at=sent_at,
    )


def _transfer(sent_at: float) -> PendingTransfer:
    return PendingTransfer(transfer=MagicMock(), sent_at=sent_at)


def test_ttl_disabled_evicts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=None,
        prefill_reqs={"r": _pending("r", 0, 0.0)},
        inflight_transfers={"r": _transfer(0.0)},
        prefill_reqs_per_replica=[1, 0],
    )
    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]
    assert "r" in self_obj.prefill_reqs
    assert "r" in self_obj.inflight_transfers
    self_obj.kv_cache.release.assert_not_called()
    self_obj.response_queue.put_nowait.assert_not_called()


def test_evicts_stuck_prefill_req(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PrefillResponse: prefill_reqs entry past TTL must be evicted."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=30.0,
        prefill_reqs={
            "stuck": _pending("stuck", 1, 900.0),  # 100s ago > 30s TTL
            "fresh": _pending("fresh", 0, 999.0),  # 1s ago
        },
        inflight_transfers={},
        prefill_reqs_per_replica=[1, 1],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "stuck" not in self_obj.prefill_reqs
    assert "fresh" in self_obj.prefill_reqs
    assert self_obj.prefill_reqs_per_replica == [1, 0]
    self_obj.kv_cache.release.assert_called_once_with("stuck", replica_idx=1)
    self_obj.response_queue.put_nowait.assert_called_once_with(
        {"stuck": SchedulerResult.cancelled()}
    )
    self_obj.dispatcher.send_request_nowait.assert_called_once()


def test_evicts_stuck_inflight_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transfer never completed: inflight_transfers + prefill_reqs evicted."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    pending_transfer = _transfer(920.0)  # 80s ago
    self_obj = _make_self(
        ttl_s=30.0,
        prefill_reqs={"stuck": _pending("stuck", 0, 950.0)},
        inflight_transfers={"stuck": pending_transfer},
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "stuck" not in self_obj.inflight_transfers
    assert "stuck" not in self_obj.prefill_reqs
    assert self_obj.prefill_reqs_per_replica == [0, 0]
    self_obj.transfer_engine.cleanup_transfer.assert_called_once_with(
        pending_transfer.transfer
    )
    self_obj.kv_cache.release.assert_called_once_with("stuck", replica_idx=0)
    self_obj.response_queue.put_nowait.assert_called_once_with(
        {"stuck": SchedulerResult.cancelled()}
    )
    self_obj.dispatcher.send_request_nowait.assert_called_once()


def test_skips_prefill_when_transfer_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old prefill_reqs.sent_at must not evict if a fresh transfer exists."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=30.0,
        prefill_reqs={"r": _pending("r", 0, 900.0)},  # 100s ago (stale)
        inflight_transfers={"r": _transfer(999.0)},  # 1s ago (fresh)
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "r" in self_obj.prefill_reqs
    assert "r" in self_obj.inflight_transfers
    assert self_obj.prefill_reqs_per_replica == [1, 0]
    self_obj.kv_cache.release.assert_not_called()
    self_obj.response_queue.put_nowait.assert_not_called()


def test_healthy_entries_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entirely fresh batch yields no evictions."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=30.0,
        prefill_reqs={"a": _pending("a", 0, 999.0)},
        inflight_transfers={"b": _transfer(998.0)},
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "a" in self_obj.prefill_reqs
    assert "b" in self_obj.inflight_transfers
    assert self_obj.prefill_reqs_per_replica == [1, 0]
    self_obj.response_queue.put_nowait.assert_not_called()
