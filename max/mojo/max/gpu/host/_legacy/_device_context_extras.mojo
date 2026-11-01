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

"""Target-specific `DeviceContext` extensions.

A plugin overlay replaces this file wholesale to add overloads for its target.
Per-target behavior that base code dispatches on belongs in the plugin's
`MaxPluginHooks` instead, so an overlay only carries what base can't express.
"""

from .._launch_args import _IdentityWrapper
from .device_context import DeviceContextBase


__extension DeviceContextBase:
    comptime FunctionWrapper = _IdentityWrapper
