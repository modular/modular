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
"""Regression tests for P2P access enablement in signal_buffers.

Regression for a bug where enable_all_peer_access() was removed from
signal_buffers(), causing:
  ValueError: Broadcast currently requires P2P access between GPUs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from max.pipelines.lib import AlwaysSignalBuffersMixin
from test_common.mocks.pipeline_config import (
    DummyPipelineConfig,
    mock_huggingface_config,
)
from test_common.mocks.pipeline_model import MockPipelineModel

_MODULE = "max.pipelines.lib.interfaces.pipeline_model"


class _AlwaysSignalModel(AlwaysSignalBuffersMixin):
    """Minimal concrete class with AlwaysSignalBuffersMixin for testing."""

    def __init__(self, devices: list) -> None:
        self.devices = devices


@mock_huggingface_config
def test_pipeline_model_signal_buffers_enables_p2p() -> None:
    """signal_buffers must call enable_all_peer_access() for multi-GPU setups.

    Regression test: a20f0fe removed this call, breaking broadcast ops on
    models such as Llama 405B and Deepseek that use multi-GPU communication.
    """
    pipeline_config = DummyPipelineConfig(
        model_path="test/model",
        quantization_encoding=MagicMock(),
        max_batch_size=1,
        max_length=128,
    )
    model = MockPipelineModel(
        pipeline_config=pipeline_config,
        session=MagicMock(),
        devices=[MagicMock(), MagicMock()],
        kv_cache_config=MagicMock(),
        weights=MagicMock(),
    )

    with (
        patch(f"{_MODULE}.enable_all_peer_access") as mock_enable_p2p,
        patch(f"{_MODULE}.is_virtual_device_mode", return_value=False),
        patch(f"{_MODULE}.Buffer") as mock_buffer,
        patch("max.nn.comm.Signals.NUM_BYTES", 1024),
    ):
        mock_buffer.zeros.return_value = MagicMock()
        _ = model.signal_buffers

    mock_enable_p2p.assert_called_once()


@mock_huggingface_config
def test_pipeline_model_signal_buffers_no_p2p_for_single_device() -> None:
    """signal_buffers returns [] without P2P enablement for single device."""
    pipeline_config = DummyPipelineConfig(
        model_path="test/model",
        quantization_encoding=MagicMock(),
        max_batch_size=1,
        max_length=128,
    )
    model = MockPipelineModel(
        pipeline_config=pipeline_config,
        session=MagicMock(),
        devices=[MagicMock()],
        kv_cache_config=MagicMock(),
        weights=MagicMock(),
    )

    with (
        patch(f"{_MODULE}.enable_all_peer_access") as mock_enable_p2p,
        patch(f"{_MODULE}.is_virtual_device_mode", return_value=False),
    ):
        result = model.signal_buffers

    assert result == []
    mock_enable_p2p.assert_not_called()


def test_always_signal_buffers_mixin_enables_p2p() -> None:
    """AlwaysSignalBuffersMixin.signal_buffers must call enable_all_peer_access() for multi-GPU."""
    model = _AlwaysSignalModel(devices=[MagicMock(), MagicMock()])

    with (
        patch(f"{_MODULE}.enable_all_peer_access") as mock_enable_p2p,
        patch(f"{_MODULE}.is_virtual_device_mode", return_value=False),
        patch(f"{_MODULE}.Buffer") as mock_buffer,
        patch("max.nn.comm.Signals.NUM_BYTES", 1024),
    ):
        mock_buffer.zeros.return_value = MagicMock()
        _ = model.signal_buffers

    mock_enable_p2p.assert_called_once()


def test_always_signal_buffers_mixin_no_p2p_for_single_device() -> None:
    """AlwaysSignalBuffersMixin skips P2P enablement for single-device models."""
    model = _AlwaysSignalModel(devices=[MagicMock()])

    with (
        patch(f"{_MODULE}.enable_all_peer_access") as mock_enable_p2p,
        patch(f"{_MODULE}.is_virtual_device_mode", return_value=False),
        patch(f"{_MODULE}.Buffer") as mock_buffer,
        patch("max.nn.comm.Signals.NUM_BYTES", 1024),
    ):
        mock_buffer.zeros.return_value = MagicMock()
        _ = model.signal_buffers

    mock_enable_p2p.assert_not_called()
