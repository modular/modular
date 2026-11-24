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

# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

# A generic variable outside the default address space has a trivially
# deinitable type, because the `where` clause makes the parser accept it. Its
# lifetime ends without a call to the destructor witness, which takes `self`
# in the default address space.


# CHECK-LABEL: lit.fn @"end_of_scope
# CHECK: lit.ref.store %{{.*}}, %s
# CHECK-NOT: __deinit__
# CHECK: lit.var.lifetime.end %s
# CHECK: hlcf.return
def end_of_scope[
    T: ImplicitlyCopyable & Deinitable
](value: T) where conforms_to(T, TrivialRegisterPassable):
    @__address_space(.SHARED)
    var s = value
    _ = s


# CHECK-LABEL: lit.fn @"reassign
# CHECK: lit.ref.store %{{.*}}, %s
# CHECK-NOT: __deinit__
# CHECK: lit.var.lifetime.end %s
# CHECK-NOT: __deinit__
# CHECK: lit.ref.store %{{.*}}, %s
# CHECK-NOT: __deinit__
# CHECK: lit.var.lifetime.end %s
# CHECK: hlcf.return
def reassign[
    T: ImplicitlyCopyable & Deinitable
](value: T) where conforms_to(T, TrivialRegisterPassable):
    @__address_space(.SHARED)
    var s = value
    s = value
    _ = s
