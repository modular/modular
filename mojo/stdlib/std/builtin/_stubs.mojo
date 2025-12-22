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

from os import abort

# ===-----------------------------------------------------------------------===#
# __MLIRType
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](ImplicitlyCopyable):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime __copyinit__is_trivial = True


# ===-----------------------------------------------------------------------===#
# @parameter for implementation details
# ===-----------------------------------------------------------------------===#


fn paramfor_has_next[
    IteratorType: ParamForIterator & Copyable
](it: IteratorType) -> Bool:
    var result = it.copy()
    try:
        _ = result.__next2__()
        return True
    except:
        return False


fn paramfor_next_iter[
    IteratorType: ParamForIterator & Copyable
](it: IteratorType) -> IteratorType:
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    try:
        _ = result.__next2__()
        return result.copy()
    except:
        abort()


fn paramfor_next_value[
    IteratorType: ParamForIterator & Copyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    try:
        var result = it.copy()
        return result.__next2__()
    except:
        abort()
