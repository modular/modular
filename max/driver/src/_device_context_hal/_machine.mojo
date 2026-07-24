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
"""Staged-tree variant of `_hal/_machine.mojo`.

In the stdlib overlay the `machine` package is staged inside `_hal`
(`gpu/host/_hal/machine`), so the re-exported names resolve relative to it.
Staged over `_hal/_machine.mojo` by `std_srcs_hal` (see
mojo/stdlib/std/BUILD.bazel); never compiled at this location.
"""

from .machine import MachineDefinition, DeviceRef, DeviceSpec
