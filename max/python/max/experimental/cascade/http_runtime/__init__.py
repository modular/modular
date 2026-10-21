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
"""HTTP-backed Cascade :py:class:`Runtime`.

.. warning::

    The pickle-over-HTTP transport is for local development and testing
    only. It has no authentication, and the server unpickles request bodies
    before validating them, so any peer that can reach it can execute
    arbitrary code with the server process's privileges. Bind only to
    loopback or a Unix socket, and never expose it to an untrusted network.
    Do not use it for anything else.
"""

from max.experimental.cascade.http_runtime.client import HttpRuntimeProxy
from max.experimental.cascade.http_runtime.subproc import SubprocHttpRuntime

__all__ = ["HttpRuntimeProxy", "SubprocHttpRuntime"]
