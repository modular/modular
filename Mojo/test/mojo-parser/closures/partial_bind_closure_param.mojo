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
# RUN: %parse-mojo-isolated %s | FileCheck %s

# Partially binding a function leaves `T` a parameter of the specialized
# generator. References to it from inside the closure trait's quoted signature
# must keep escaping the quote in the argument types too, not just in the
# parameter list.


def takes_closure[T: Movable, U: AnyType](f: Some[def(Int) -> T]):
    pass


# CHECK-LABEL: lit.fn @"call_partially_bound()"
def call_partially_bound():
    # CHECK: lit.alias.decl *"bound`": {{.*}}"f.T`": trait<{{.*}} *(1,0), mut *[0,1]>>]{{.*}}("f": !lit.ref<:trait<{{.*}} *(1,0), mut *[0,1]>>]
    comptime bound = takes_closure[_, Int]
    bound(lambda (i: Int) -> Int: i)
