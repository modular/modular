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
"""Tests for num_heads_for_device in max.nn.attention."""

from __future__ import annotations

import pytest
from max.nn.attention import num_heads_for_device


class TestNumHeadsForDevice:
    """Tests for distributing attention heads across devices."""

    def test_even_split(self) -> None:
        """Heads divide evenly across devices."""
        assert (
            num_heads_for_device(num_heads=8, device_idx=0, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=1, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=2, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=8, device_idx=3, num_devices=4) == 2
        )

    def test_remainder_goes_to_earlier_devices(self) -> None:
        """When heads don't divide evenly, earlier devices get one extra."""
        # 7 heads across 4 devices: 2, 2, 2, 1
        assert (
            num_heads_for_device(num_heads=7, device_idx=0, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=1, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=2, num_devices=4) == 2
        )
        assert (
            num_heads_for_device(num_heads=7, device_idx=3, num_devices=4) == 1
        )

    def test_single_device(self) -> None:
        """Single device gets all heads."""
        assert (
            num_heads_for_device(num_heads=16, device_idx=0, num_devices=1)
            == 16
        )

    def test_one_head_per_device(self) -> None:
        """Each device gets exactly one head."""
        assert (
            num_heads_for_device(num_heads=4, device_idx=0, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=1, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=2, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=4, device_idx=3, num_devices=4) == 1
        )

    def test_more_devices_than_heads(self) -> None:
        """Excess devices get zero heads."""
        # 2 heads across 4 devices: 1, 1, 0, 0
        assert (
            num_heads_for_device(num_heads=2, device_idx=0, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=1, num_devices=4) == 1
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=2, num_devices=4) == 0
        )
        assert (
            num_heads_for_device(num_heads=2, device_idx=3, num_devices=4) == 0
        )

    @pytest.mark.parametrize("num_heads", [1, 7, 16, 31, 64])
    @pytest.mark.parametrize("num_devices", [1, 2, 3, 4, 8])
    def test_total_across_all_devices_equals_num_heads(
        self, num_heads: int, num_devices: int
    ) -> None:
        """Sum of heads across all devices equals the total."""
        total = sum(
            num_heads_for_device(
                num_heads=num_heads,
                device_idx=i,
                num_devices=num_devices,
            )
            for i in range(num_devices)
        )
        assert total == num_heads
