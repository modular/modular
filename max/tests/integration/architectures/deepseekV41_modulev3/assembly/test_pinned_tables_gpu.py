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
"""GPU tests for the model's page-locked Engram host tables.

``_pinned_table`` is the allocation every decode step's host-table gather
reads over UVA: the tables must be page-locked against the accelerator (a
``CPU()`` staging buffer is pageable, which the gather must not read),
byte-exact against the checkpoint view, and shared across reloads of the
same bytes.
"""

from __future__ import annotations

import hashlib
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.model import _pinned_table

ROWS = 5


def _codes() -> np.ndarray:
    return np.arange(ROWS * 256, dtype=np.uint8).reshape(ROWS, 256)


def test_pinned_table_is_page_locked_and_byte_exact() -> None:
    device = Accelerator()
    codes = _codes()
    table = _pinned_table(
        device,
        f"engram_test_{uuid4().hex}_weight",
        ROWS,
        256,
        codes,
        DType.float8_e4m3fn,
    )
    assert table.pinned
    assert table.dtype == DType.float8_e4m3fn
    assert [int(dim) for dim in table.shape] == [ROWS, 256]
    np.testing.assert_array_equal(table.view(DType.uint8).to_numpy(), codes)


def test_pinned_table_dedupes_reloads_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reload dedupe hashes only once a same-shaped table exists.

    The first load of a shape has nothing to compare against: hashing a
    production table streams ~98 GB on one thread, so it must not run. A
    reload hashes to prove its bytes match, then shares the pin.
    """
    device = Accelerator()
    name = f"engram_test_{uuid4().hex}_weight"
    codes = _codes()
    digest_calls = Mock(wraps=hashlib.blake2b)
    monkeypatch.setattr(hashlib, "blake2b", digest_calls)
    first = _pinned_table(device, name, ROWS, 256, codes, DType.float8_e4m3fn)
    digest_calls.assert_not_called()
    second = _pinned_table(device, name, ROWS, 256, codes, DType.float8_e4m3fn)
    # The reload hashes the incoming bytes and the first pin's own bytes
    # (its digest was deferred), then shares the allocation.
    assert digest_calls.call_count == 2
    assert second._data_ptr() == first._data_ptr()
    # Different bytes under the same identity pin their own copy.
    third = _pinned_table(
        device, name, ROWS, 256, codes + 1, DType.float8_e4m3fn
    )
    assert third._data_ptr() != first._data_ptr()
    np.testing.assert_array_equal(third.view(DType.uint8).to_numpy(), codes + 1)
