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

"""Distributed KV cache connector via dKV service.

Includes the vendored dKV Python client (protocol, ZMQ transport)
and the MAX KVConnector implementation.
"""

from .client import DKVClient
from .connector import DKVConnector, DKVExternalBlockMetadata
from .protocol import BlockDescriptor

__all__ = [
    "BlockDescriptor",
    "DKVClient",
    "DKVConnector",
    "DKVExternalBlockMetadata",
]
