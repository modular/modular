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
"""Test the max.driver.DeviceEvent Python bindings."""

import re

import pytest
from max import driver
from max.driver import DeviceEvent


@pytest.fixture
def device() -> driver.Accelerator:
    """Fixture to provide an accelerator device."""
    if not driver.accelerator_count():
        pytest.skip("Requires GPU")
    return driver.Accelerator()


def test_device_event_str_and_repr(device: driver.Accelerator) -> None:
    """Test DeviceEvent __str__ and __repr__ methods."""
    event = DeviceEvent(device)

    # __str__ should return format: {api}[{device_id}]:event@{pointer}
    # e.g., "cuda[0]:event@0x7f8b40000000"
    event_str = str(event)

    # First check it's not empty
    assert len(event_str) > 0, "String representation should not be empty"

    # Validate format with regex: {api}[{device_id}]:event@{pointer}
    pattern = rf"^{re.escape(device.api)}\[\d+\]:event@(0x)?[0-9a-f]+$"
    assert re.match(pattern, event_str), (
        f"String '{event_str}' should match pattern '{pattern}'"
    )

    # __repr__ should be identical to __str__
    assert repr(event) == event_str, "__repr__ should be identical to __str__"

    # Different events should have different pointer addresses
    event2 = DeviceEvent(device)
    event2_str = str(event2)

    pointer1 = event_str.split(":event@")[1]
    pointer2 = event2_str.split(":event@")[1]
    assert pointer1 != pointer2, (
        "Different events should have different pointer addresses"
    )
