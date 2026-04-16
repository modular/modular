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
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllWritableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Writable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Writable`.
"""

comptime _AllWritableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Writable) and Prev

comptime AllMovable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllMovableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Movable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Movable`.
"""

comptime _AllMovableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Movable) and Prev

comptime AllCopyable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllCopyableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Copyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Copyable`.
"""

comptime _AllCopyableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Copyable) and Prev

comptime AllImplicitlyCopyable[
    *Ts: AnyType
]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllImplicitlyCopyableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyCopyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyCopyable`.
"""

comptime _AllImplicitlyCopyableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], ImplicitlyCopyable) and Prev

comptime AllDefaultable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllDefaultableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Defaultable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Defaultable`.
"""

comptime _AllDefaultableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Defaultable) and Prev

comptime AllEquatable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllEquatableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Equatable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Equatable`.
"""

comptime _AllEquatableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Equatable) and Prev

comptime AllHashable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllHashableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `Hashable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Hashable`.
"""

comptime _AllHashableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], Hashable) and Prev

comptime AllImplicitlyDestructible[
    *Ts: AnyType
]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllImplicitlyDestructibleReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyDestructible`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyDestructible`.
"""

comptime _AllImplicitlyDestructibleReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], ImplicitlyDestructible) and Prev

comptime AllRegisterPassable[*Ts: AnyType]: Bool = _ReduceVariadicAndIdxToValue[
    BaseVal=True,
    ParamListType=Ts.values,
    Reducer=_AllRegisterPassableReducer,
]
"""Evaluates to `True` if all types in `Ts` conform to `RegisterPassable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `RegisterPassable`.
"""

comptime _AllRegisterPassableReducer[
    Prev: Bool,
    From: Variadic.TypesOfTrait[AnyType],
    idx: SIMDSize,
] = conforms_to(From[idx], RegisterPassable) and Prev
