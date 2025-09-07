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
struct __MLIRType[T: AnyTrivialRegType](ImplicitlyCopyable, Movable):
    var value: T


# ===-----------------------------------------------------------------------===#
# @parameter for implementation details
# ===-----------------------------------------------------------------------===#


fn paramfor_next_iter[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ will return true.  This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    _ = result.__next__()
    return result.copy()


fn paramfor_next_value[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ will return true.  This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    return result.__next__()
