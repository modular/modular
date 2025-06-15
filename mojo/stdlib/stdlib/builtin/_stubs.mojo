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


# ===-----------------------------------------------------------------------===#
# __MLIRType
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](Copyable, ExplicitlyCopyable, Movable):
    var value: T

    fn copy(self) -> Self:
        return self


# ===-----------------------------------------------------------------------===#
# parameter_for
# ===-----------------------------------------------------------------------===#


trait _ParamForIterator(Copyable):
    alias _IndexType: Copyable

    fn __has_next__(self) -> Bool:
        ...

    fn __next__(mut self) -> _IndexType:
        ...


@fieldwise_init
struct _ParamForIteratorWrapper[IteratorT: _ParamForIterator]:
    var next_it: IteratorT
    var value: IteratorT._IndexType


fn parameter_for_generator[
    IteratorT: _ParamForIterator
](it: IteratorT) -> _ParamForIteratorWrapper[IteratorT]:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ returns true.
    var next_it = it
    return _ParamForIteratorWrapper(next_it, next_it.__next__())
