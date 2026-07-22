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

"""CPU conformance tests for the ``KVConnector`` protocol additions.

Covers the ``wait_for_loads`` / ``wait_for_offloads`` barriers, the ``offload``
``parent_seq_hash`` parameter, and the fire-and-forget ``touch`` method,
verified against the no-op ``NullConnector`` (which needs no device) plus the
host/disk connectors.
"""

from __future__ import annotations

from max.pipelines.kv_cache.connectors import (
    LocalConnector,
    NullConnector,
    TieredConnector,
)
from max.pipelines.kv_cache.kv_connector import KVConnector


def test_null_connector_satisfies_protocol() -> None:
    assert isinstance(NullConnector(), KVConnector)


def test_barrier_methods_are_callable() -> None:
    connector = NullConnector()
    # Both are no-op barriers (return ``None``); just ensure they are callable.
    connector.wait_for_loads()
    connector.wait_for_offloads()


def test_offload_accepts_parent_seq_hash() -> None:
    # The new third positional/keyword arg is accepted (and ignored here).
    NullConnector().offload([0], [b"\x01" * 8], parent_seq_hash=b"\x02" * 8)
    # ``None`` is the root-of-chain sentinel under the bytes-only contract.
    NullConnector().offload([0], [b"\x01" * 8], parent_seq_hash=None)


def test_touch_returns_none_and_never_raises() -> None:
    # ``touch`` is fire-and-forget with a ``-> None`` annotation, so just call
    # it: accepts ``replica_idx``, tolerates an empty payload, never raises.
    connector = NullConnector()
    connector.touch([b"\x01" * 8])
    connector.touch([b"\x01" * 8, b"\x02" * 8], replica_idx=1)
    connector.touch([])

    # ``LocalConnector`` / ``TieredConnector`` need device buffers to fully
    # construct (see the GPU connector tests), but ``touch`` ignores instance
    # state, so exercise it on an uninitialized instance -- the same ``__new__``
    # pattern used in ``test_kv_connector_reset_metrics.py`` -- to prove it is a
    # no-op that never raises.
    local_connector = LocalConnector.__new__(LocalConnector)
    local_connector.touch([b"\x01" * 8])
    local_connector.touch([b"\x01" * 8], replica_idx=2)

    tiered_connector = TieredConnector.__new__(TieredConnector)
    tiered_connector.touch([b"\x01" * 8])
    tiered_connector.touch([b"\x01" * 8], replica_idx=2)


def test_touch_preserves_protocol_conformance() -> None:
    # Adding ``touch`` to the Protocol must not break structural conformance.
    # ``NullConnector`` is CPU-constructable, so assert the full runtime check;
    # for the host/disk connectors ``@runtime_checkable`` ``isinstance`` invokes
    # their state-reading properties (GPU-bound to construct), so assert the new
    # Protocol member is present at the class level instead.
    assert isinstance(NullConnector(), KVConnector)
    assert hasattr(LocalConnector, "touch")
    assert hasattr(TieredConnector, "touch")
