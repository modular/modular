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

from builtin.range import _StridedRangeIterator, _UIntStridedRangeIterator

# ===-----------------------------------------------------------------------===#
# __MLIRType
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](Movable, Copyable, ExplicitlyCopyable):
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


struct _ParamForIteratorWrapper[IteratorT: _ParamForIterator]:
    var next_it: IteratorT
    var value: IteratorT._IndexType
    var stop: Bool

    fn __init__(
        out self,
        next_it: IteratorT,
        owned value: IteratorT._IndexType,
        stop: Bool,
    ):
        self.next_it = next_it
        self.value = value^
        self.stop = stop


fn parameter_for_generator[
    IteratorT: _ParamForIterator
](it: IteratorT) -> _ParamForIteratorWrapper[IteratorT]:
    if it.__has_next__():
        var next_it = it
        return _ParamForIteratorWrapper(next_it, next_it.__next__(), False)

    var next_iter: IteratorT
    __mlir_op.`lit.ownership.mark_initialized`(
        __get_mvalue_as_litref(next_iter)
    )
    var next_val: IteratorT._IndexType
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(next_val))
    return _ParamForIteratorWrapper(next_iter^, next_val^, True)
