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

"""Unit tests for the KV cache host capacity preflight."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.pipelines.kv_cache.paged_kv_cache import block_copy_engine


def test_host_capacity_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_copy_engine.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1024),
    )

    with pytest.raises(RuntimeError, match="host_kvcache_swap_space_gb"):
        block_copy_engine._check_host_memory_capacity(2048)


def test_host_capacity_accepts_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        block_copy_engine.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    block_copy_engine._check_host_memory_capacity(4096)


def test_host_capacity_skips_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _raise() -> None:
        raise OSError("host memory unavailable")

    monkeypatch.setattr(block_copy_engine.psutil, "virtual_memory", _raise)

    block_copy_engine._check_host_memory_capacity(1 << 60)
    assert "skipping KV cache host capacity preflight" in caplog.text
