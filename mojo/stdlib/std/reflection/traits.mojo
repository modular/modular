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
"""Compile-time meta functions for checking trait conformance across variadic type lists.
"""

from std.builtin.variadics import _ReduceVariadicAndIdxToValue

comptime AllWritable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllWritableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Writable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Writable`.
"""

comptime _AllWritableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Writable) and Prev[0]]

comptime AllMovable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllMovableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Movable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Movable`.
"""

comptime _AllMovableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Movable) and Prev[0]]

comptime AllCopyable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllCopyableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Copyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Copyable`.
"""

comptime _AllCopyableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Copyable) and Prev[0]]

comptime AllImplicitlyCopyable[
    *Ts: AnyType
]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllImplicitlyCopyableReducer,
][
    0
]
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyCopyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyCopyable`.
"""

comptime _AllImplicitlyCopyableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], ImplicitlyCopyable) and Prev[0]]

comptime AllDefaultable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllDefaultableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Defaultable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Defaultable`.
"""

comptime _AllDefaultableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Defaultable) and Prev[0]]

comptime AllEquatable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllEquatableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Equatable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Equatable`.
"""

comptime _AllEquatableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Equatable) and Prev[0]]

comptime AllHashable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllHashableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `Hashable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Hashable`.
"""

comptime _AllHashableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], Hashable) and Prev[0]]

comptime AllImplicitlyDestructible[
    *Ts: AnyType
]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllImplicitlyDestructibleReducer,
][
    0
]
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyDestructible`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyDestructible`.
"""

comptime _AllImplicitlyDestructibleReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], ImplicitlyDestructible) and Prev[0]]

comptime AllRegisterPassable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=Variadic.values[True],
    VariadicType=Ts,
    Reducer=_AllRegisterPassableReducer,
][0]
"""Evaluates to `True` if all types in `Ts` conform to `RegisterPassable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `RegisterPassable`.
"""

comptime _AllRegisterPassableReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = Variadic.values[conforms_to(From[idx], RegisterPassable) and Prev[0]]
