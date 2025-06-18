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


# This type is tightly bound to the internals of "@parameter for" emission.
struct _ParamForWrapper[Iter: IteratorTrait & Copyable]:
    var next_it: Iter
    var value: Iter.Element

    fn __init__(out self, it: Iter):
        self.next_it = it
        self.value = self.next_it.__next__()


fn parameter_for_generator[
    Iter: IteratorTrait & Copyable
](it: Iter) -> _ParamForWrapper[Iter]:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ returns true.
    return _ParamForWrapper(it)
