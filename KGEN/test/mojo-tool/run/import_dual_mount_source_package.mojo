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

# A package's identity is its position, so one directory bound under two names
# is two packages, and every type it declares exists twice over. Overlapping
# import roots are the easy way to reach that: here `lib` and `lib/mylib` both
# name the same `utils`. Reported at the second binding, rather than left to
# surface later as a mismatch between two types that print identically.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/mylib/utils
# RUN: echo "# pkg" > %t.dir/lib/mylib/utils/__init__.mojo
# RUN: echo "struct Thing:" > %t.dir/lib/mylib/utils/a.mojo
# RUN: echo "    var x: Int" >> %t.dir/lib/mylib/utils/a.mojo
# RUN: echo "" >> %t.dir/lib/mylib/utils/a.mojo
# RUN: echo "    def __init__(out self):" >> %t.dir/lib/mylib/utils/a.mojo
# RUN: echo "        self.x = 7" >> %t.dir/lib/mylib/utils/a.mojo
# RUN: not mojo run -I %t.dir/lib -I %t.dir/lib/mylib %s 2>&1 | FileCheck %s

# CHECK: error: package imported as 'mylib.utils' must not also be imported as 'utils'; remove the duplicate import root or file that reaches it twice

from mylib.utils.a import Thing
from utils.a import Thing as SameThing


def main():
    var t = Thing()
    print(t.x)
