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

"""Public Python surface for the Rust host/disk tiered KV connector.

``TierConnector`` and ``TierTransfer`` are implemented in Rust and exposed by
the ``kv_tier_connector._kv_tier_connector_pyo3`` extension module. This shim
re-exports them so callers ``from kv_tier_connector import TierConnector``.
"""

from __future__ import annotations

from ._kv_tier_connector_pyo3 import (  # type: ignore[import-not-found]
    TierConnector,
    TierTransfer,
)

__all__ = ["TierConnector", "TierTransfer"]
