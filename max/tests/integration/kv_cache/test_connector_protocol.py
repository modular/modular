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

Covers the ``wait_for_loads`` / ``wait_for_offloads`` barriers and the
``offload`` ``parent_seq_hash`` parameter, verified against the no-op
``NullConnector`` (which needs no device).
"""

from __future__ import annotations

from max.pipelines.kv_cache.connectors import NullConnector
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
