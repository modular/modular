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

# As for a source package: one artifact bound under two names is two packages,
# with two of every type it declares. A second name for one artifact need not
# come from a second import root, so this reaches it with a link instead: both
# names sit at the same root and resolve to one file.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/utils %t.dir/lib
# RUN: echo "# pkg" > %t.dir/src/utils/__init__.mojo
# RUN: echo "struct Thing:" > %t.dir/src/utils/a.mojo
# RUN: echo "    var x: Int" >> %t.dir/src/utils/a.mojo
# RUN: echo "" >> %t.dir/src/utils/a.mojo
# RUN: echo "    def __init__(out self):" >> %t.dir/src/utils/a.mojo
# RUN: echo "        self.x = 7" >> %t.dir/src/utils/a.mojo
# RUN: echo "" >> %t.dir/src/utils/a.mojo
# RUN: echo "def make() -> Thing:" >> %t.dir/src/utils/a.mojo
# RUN: echo "    return Thing()" >> %t.dir/src/utils/a.mojo
# RUN: mojo precompile %t.dir/src/utils -o %t.dir/lib/utils.mojoc
# RUN: rm -rf %t.dir/src
# RUN: ln -s utils.mojoc %t.dir/lib/dup.mojoc
# RUN: not mojo run -I %t.dir/lib %s 2>&1 | FileCheck %s

# CHECK: error: package imported as 'utils' must not also be imported as 'dup'; remove the duplicate import root or file that reaches it twice

from utils.a import Thing
from dup.a import make


def main():
    var t: Thing = make()
    print(t.x)
