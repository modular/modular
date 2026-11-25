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
# Generic code that requires a route rejects an engine without it: the
# `tcgen05.ld` engine counts its loads and has no groups.

from layout.atoms import HasGroups
from layout.atoms.tmem import TMemLoad


def drain[E: HasGroups]():
    E.Groups.Commit()()
    E.Groups.Wait()()


def main():
    # CHECK: 'drain' parameter 'E' has 'HasGroups' type, but value has type 'AnyStruct[Tcgen05LdEngine]'
    drain[TMemLoad[.float32, 16].AsyncEngine]()
