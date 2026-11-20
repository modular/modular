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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# A prepended implicit parameter that depends on non-inferred parameters is
# diagnosed once per offending parameter, however often its type mentions it.


struct Pair[a: Int, b: Int](Movable where False):
    pass


struct NeedsPair[a: Int, b: Int, p: Pair[a, b]](Movable where False):
    pass


# CHECK: [[@LINE+2]]:{{[0-9]+}}: error: inferred parameter of type 'Pair[c, c]' cannot depend on non-inferred parameter 'c'
# CHECK-NOT: non-inferred parameter 'c'
struct RepeatedDep[c: Int, X: NeedsPair[c, c, _]]:
    pass


# CHECK: [[@LINE+3]]:{{[0-9]+}}: error: inferred parameter of type 'Pair[a, b]' cannot depend on non-inferred parameter 'a'
# CHECK: [[@LINE+2]]:{{[0-9]+}}: error: inferred parameter of type 'Pair[a, b]' cannot depend on non-inferred parameter 'b'
# CHECK-NOT: cannot depend on non-inferred parameter
struct TwoDeps[a: Int, b: Int, X: NeedsPair[a, b, _]]:
    pass


# The element type message doesn't name the offending parameter, so it is
# reported once even when several parameters are at fault.
# CHECK: [[@LINE+2]]:{{[0-9]+}}: error: element type parameter 'NeedsPair[a, b, p]' must be an 'inferred' parameter
# CHECK-NOT: element type parameter
def variadic_deps[a: Int, b: Int, *Xs: NeedsPair[a, b, _]]():
    pass
