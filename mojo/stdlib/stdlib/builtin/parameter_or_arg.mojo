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

from testing import assert_true, assert_equal


struct Parameter[
    T: Copyable & Movable, //, value:T
](Copyable, Movable):
    fn __init__(
        out self,
    ):
        pass


struct ParameterOrArg[
    T: Copyable & Movable, //, _comptime_value: Optional[T] , 
](Copyable, Movable):
    alias is_parameter = Self._comptime_value is not None
    alias _arr_size = 0 if Self.is_parameter else 1
    var _runtime_value: InlineArray[Self.T, Self._arr_size]
    """Call this to get the value if it's known at compile time. 
    
    If you try to call it but the value is know at runtime, the compilation will fail.
    """
    alias comptime_value = Self._comptime_value.value()

    @implicit
    fn __init__(
        out self: ParameterOrArg[
             _comptime_value = Optional[T](None)
        ],
        owned value: Self.T,
    ):
        self._runtime_value = InlineArray[Self.T, self._arr_size](value)

    @implicit
    fn __init__[
        V: Self.T
    ](
        out self: ParameterOrArg[
            _comptime_value = Optional[T](V)
        ],
        value: Parameter[V],
    ):
        self._runtime_value = InlineArray[Self.T, self._arr_size](uninitialized=True)

    @always_inline
    fn runtime_value(self) -> T:
        """Can only be used if the value is known at runtime, otherwise the compilation will fail.
        """
        constrained[
            not self.is_parameter,
            (
                "You tried to call the runtime_value but it's known at compile"
                " time. If you don't really care if it's comptime or runtime,"
                " use .value()."
            ),
        ]()
        return self._runtime_value[0]

    @always_inline
    fn value(self) -> T:
        """Can be used if the value is known at compile time or runtime."""

        @parameter
        if self.is_parameter:
            return self.comptime_value
        else:
            return self._runtime_value[0]
