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
"""Pins the inputs the Hy4 target adapter adds past the canonical tail."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.hy_v4.spec_adapters import (
    ADAPTER,
    BATCH_CONTEXT_LENGTH,
)
from max.pipelines.speculative import spec_adapters


def _context(current_position: int) -> Any:
    return SimpleNamespace(
        tokens=SimpleNamespace(current_position=current_position)
    )


def test_is_the_registered_adapter() -> None:
    assert spec_adapters.primary_kv("HYV4ForCausalLM", "mtp") == "mla"


def test_takes_the_host_prefill_buffer_length() -> None:
    (length,) = ADAPTER.trailing_input_types(None)
    assert length.dtype == DType.int32
    assert list(length.shape) == [1]
    assert length.device == DeviceRef.CPU()


def test_prefill_buffer_length_rounds_each_request_up_to_a_page() -> None:
    config = cast(
        Any, SimpleNamespace(kv_params=SimpleNamespace(page_size=128))
    )
    (length,) = ADAPTER.trailing_buffers(
        config, [_context(1), _context(128), _context(129)], []
    )
    assert length.to_numpy().tolist() == [128 + 128 + 256]


def test_names_the_prefill_buffer_length() -> None:
    value = SimpleNamespace(tensor="length")
    assert ADAPTER.extra(None, [cast(Any, value)]) == {
        BATCH_CONTEXT_LENGTH: "length"
    }
