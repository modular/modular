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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

# Tests for generic types that constrain their own Writable implementation.

from memory import ArcPointer, OwnedPointer
from sys import env_get_int


@fieldwise_init
struct NotWritable(ImplicitlyCopyable):
    pass


def main():
    var string = String()

    comptime if env_get_int["test"]() == 1:
        # CHECK_1: constraint failed: {{.*}}ArcPointer{{.*}}(Writable) conformance requires {{.*}}NotWritable{{.*}} conformance, which is not satisfied.
        var ptr = ArcPointer[NotWritable](NotWritable())
        string.write(ptr)
    elif env_get_int["test"]() == 2:
        # CHECK_2: constraint failed: {{.*}}OwnedPointer{{.*}}(Writable) conformance requires {{.*}}NotWritable{{.*}} conformance, which is not satisfied.
        var ptr = OwnedPointer[NotWritable](NotWritable())
        string.write(ptr)
