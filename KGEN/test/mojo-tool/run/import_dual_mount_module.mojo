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

# A single module file is no different from a package here: identity is
# position, so one file bound under two names declares two of every type in it.
# `lib` and `lib/mylib` both reach the same `utils.mojo`.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/lib/mylib
# RUN: echo "struct Thing:" > %t.dir/lib/mylib/utils.mojo
# RUN: echo "    var x: Int" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "    def __init__(out self):" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "        self.x = 7" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "def make() -> Thing:" >> %t.dir/lib/mylib/utils.mojo
# RUN: echo "    return Thing()" >> %t.dir/lib/mylib/utils.mojo
# RUN: not mojo run -I %t.dir/lib -I %t.dir/lib/mylib %s 2>&1 | FileCheck %s

# CHECK: error: module imported as 'mylib.utils' must not also be imported as 'utils'; remove the duplicate import root or file that reaches it twice

# Both bindings are used: an unused import is never resolved, so nothing would
# bind the module a second time.
from mylib.utils import Thing
from utils import make


def main():
    var t: Thing = make()
    print(t.x)
