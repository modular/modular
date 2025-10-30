# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Unit tests for DeviceRef hashing and equality semantics."""

from __future__ import annotations

from max.graph import DeviceRef


def test_device_ref_hashable_and_equality() -> None:
    # GPU device refs with same identity should be equal and hash-identical.
    d0 = DeviceRef.GPU(0)
    d0_b = DeviceRef.GPU(0)
    assert d0 == d0_b
    assert hash(d0) == hash(d0_b)

    # Distinct devices should not be equal and should hash differently often.
    d1 = DeviceRef.GPU(1)
    assert hash(d0) != hash(d1)
    assert d0 != d1

    # Dict key usage must work via equality semantics.
    m = {d0: "gpu0"}
    assert m[d0_b] == "gpu0"

    m[d1] = "gpu1"
    assert m[DeviceRef.GPU(1)] == "gpu1"


def test_device_ref_hashable_cpu() -> None:
    c0 = DeviceRef.CPU(0)
    c0_b = DeviceRef.CPU(0)
    c1 = DeviceRef.CPU(1)

    assert c0 == c0_b
    assert hash(c0) == hash(c0_b)
    assert c0 != c1

    s = {c0, c1}
    assert c0_b in s
