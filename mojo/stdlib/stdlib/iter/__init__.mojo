# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


trait Iterable:
    """The `Iterator` trait describes a type that can be turned into an
    iterator.
    """

    alias IteratorType[mut: Bool, //, origin: Origin[mut]]: Iterator

    fn __iter__(ref self) -> Self.IteratorType[__origin_of(self)]:
        ...


trait Iterator(Movable):
    """The `Iterator` trait describes a type that can be used as an
    iterator, e.g. in a `for` loop.
    """

    alias Element: ExplicitlyCopyable & Movable

    fn __has_next__(self) -> Bool:
        ...

    fn __next__(mut self) -> Self.Element:
        ...


@always_inline
fn iter[
    IterableType: Iterable
](ref iterable: IterableType) -> IterableType.IteratorType[
    __origin_of(iterable)
]:
    return iterable.__iter__()


@always_inline
fn next[
    IteratorType: Iterator
](mut iterator: IteratorType) -> IteratorType.Element:
    return iterator.__next__()
