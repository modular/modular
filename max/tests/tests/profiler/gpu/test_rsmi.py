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

import pytest
from max.profiler.gpu._rsmi import RSMIContext, RSMIError


class _FakeLibrary:
    """Fake ROCm-SMI library: memory busy answers with a fixed status."""

    def __init__(self, busy_status: int) -> None:
        self.busy_status = busy_status

    def rsmi_status_string(self, status: int, out: object) -> int:
        return 0

    def rsmi_dev_memory_busy_percent_get(self, index: int, out: object) -> int:
        return self.busy_status


def _ctx(busy_status: int) -> RSMIContext:
    ctx = RSMIContext()
    ctx._library = _FakeLibrary(busy_status)  # type: ignore[assignment]
    return ctx


def test_memory_busy_not_supported_is_none() -> None:
    # An integrated GPU (gfx1036) answers RSMI_STATUS_NOT_SUPPORTED (2).
    assert _ctx(2)._get_memory_busy_percent(0) is None


def test_memory_busy_other_errors_propagate() -> None:
    with pytest.raises(RSMIError):
        _ctx(1)._get_memory_busy_percent(0)  # RSMI_STATUS_INVALID_ARGS
