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
# Tests that loading a BFloat16 filled with 0xFF (host poison) triggers abort.

from std.memory import UnsafePointer


# CHECK: use of uninitialized memory
def main():
    var value = UInt16(0xFFFF)
    var ptr = UnsafePointer(to=value).bitcast[BFloat16]()
    _ = ptr.load()
