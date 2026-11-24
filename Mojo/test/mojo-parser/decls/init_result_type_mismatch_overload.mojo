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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# FIXME: This code should not be compile, yet it can.


struct Lit(Intable, TrivialRegisterPassable):
    def __int__(self) -> Int:
        return 1


struct Len(TrivialRegisterPassable):
    @implicit
    def __init__(out self, value: Lit):
        pass


struct Vec[n: Int]:
    @implicit
    def __init__(out self, value: Len):
        pass

    def __init__[T: Intable](out self: Vec[1], value: T):
        pass


def splat[n: Int](x: Lit) -> Vec[n]:
    return Vec[n](x)
