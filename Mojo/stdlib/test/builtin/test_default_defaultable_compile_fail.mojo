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

# Test that the default Defaultable implementation produces a clear error
# message when a field does not implement Defaultable.


struct NotDefaultable:
    var x: Int


struct HasBadField(Defaultable):
    var field: NotDefaultable


# CHECK: constraint failed: Could not derive Defaultable for HasBadField - member field `field: NotDefaultable` does not implement Defaultable
def main() raises:
    var a = HasBadField()
    _ = a
