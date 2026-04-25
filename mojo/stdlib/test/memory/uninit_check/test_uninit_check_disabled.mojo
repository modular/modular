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
# Tests that the poison check is disabled by default
# (without -D MOJO_STDLIB_SIMD_UNINIT_CHECK).
# Loading poison values should NOT crash when the check is disabled.

from std.memory import UnsafePointer


def test_poison_ignored_when_disabled():
    """With MOJO_STDLIB_SIMD_UNINIT_CHECK not set, loading poison should
    not abort."""
    var value = UInt32(0xFFFFFFFF)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def test_device_poison_ignored_when_disabled():
    """With MOJO_STDLIB_SIMD_UNINIT_CHECK not set, loading device poison
    should not abort."""
    var value = UInt32(0x7FC00000)
    var ptr = UnsafePointer(to=value).bitcast[Float32]()
    _ = ptr.load()


def main():
    test_poison_ignored_when_disabled()
    test_device_poison_ignored_when_disabled()
