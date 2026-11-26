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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s
#
# A commit atom is linear: issuing it is the only thing that consumes it, so
# a seal built with its barrier cannot be dropped before it is issued.

from std.memory import unsafe_stack_allocation

from layout.atoms.cp_async import CpAsync
from layout.tma_async import SharedMemBarrier


def main():
    var barrier = unsafe_stack_allocation[
        1, SharedMemBarrier, address_space=.SHARED
    ]()
    comptime E = CpAsync[.float32, 16].AsyncEngine
    # CHECK: 'seal' abandoned without being explicitly destroyed: type 'CpAsyncMBarrierArrive' does not conform to 'Deinitable'
    var seal = E.Barrier.Seal[MutUntrackedOrigin](barrier)
