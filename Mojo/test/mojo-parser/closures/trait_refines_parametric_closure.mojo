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


struct MyStruct[n: Int]:
    pass


# The parameter appears in the result type.
trait T1(def[n: Int]() -> MyStruct[n]):
    pass


# The parameter appears in an argument type.
trait T2(def[n: Int](x: MyStruct[n]) -> Int):
    pass


# expected-error @below {{'S1' does not implement all requirements for 'def[n: Int]() -> MyStruct[n]'}}
# expected-note @+1 {{required function '__call__' is not implemented}}
struct S1(T1):
    pass


# expected-error @below {{'S2' does not implement all requirements for 'def[n: Int](x: MyStruct[n]) -> Int'}}
# expected-note @+1 {{required function '__call__' is not implemented}}
struct S2(T2):
    pass


def main():
    pass
