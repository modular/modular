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

# A named `Origin` parameter and an `origin_of(...)` expression that names the
# same origin are the same value, so types that differ only in which of the two
# they spell must match.


trait MyIterable:
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: AnyType

    def iter(ref self) -> Self.IteratorType[origin_of(self)]:
        ...

    def iter_named[o: ImmOrigin](ref[o] self) -> Self.IteratorType[o]:
        ...


# The callee spells the origin `origin_of(self)`; the caller names it.
def named_result[
    origin: ImmOrigin, T: MyIterable
](ref[origin] iterable: T) -> T.IteratorType[origin]:
    return iterable.iter()


# The callee names the origin; the caller spells it `origin_of(iterable)`.
def origin_of_result[
    origin: ImmOrigin, T: MyIterable
](ref[origin] iterable: T) -> T.IteratorType[origin_of(iterable)]:
    return iterable.iter_named()


# Different origins are different values, however they are spelled.
def different_origin[
    a: ImmOrigin, b: ImmOrigin, T: MyIterable
](ref[a] iterable: T) -> T.IteratorType[b]:
    # expected-error @below {{cannot implicitly convert}}
    return iterable.iter()
