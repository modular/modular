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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Verifies that `OwnedDLHandle.get_function`'s Mojo-ABI path rejects
# multi-field struct arguments at comptime. The Mojo-ABI path can
# silently corrupt struct args/returns (Mojo vs C ABI disagreement on
# aggregate passing), so the comptime assert in `_DLCallable.__call__`
# refuses to compile such calls.

from std.ffi import OwnedDLHandle


@fieldwise_init
struct _TwoFields(Copyable, Movable):
    var a: Int32
    var b: Int32


def main() raises:
    var lib = OwnedDLHandle()
    var f = lib.get_function[Int32]("ignored_symbol")
    # CHECK: aggregate (multi-field) argument types are unsafe through the Mojo ABI
    _ = f(_TwoFields(1, 2))
