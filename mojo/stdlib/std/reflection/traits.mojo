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

comptime AllWritable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsWritablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Writable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Writable`.
"""

comptime _IsWritablePredicate[T: AnyType]: Bool = conforms_to(T, Writable)

comptime AllMovable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsMovablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Movable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Movable`.
"""

comptime _IsMovablePredicate[T: AnyType]: Bool = conforms_to(T, Movable)

comptime AllCopyable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsCopyablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Copyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Copyable`.
"""

comptime _IsCopyablePredicate[T: AnyType]: Bool = conforms_to(T, Copyable)

comptime AllImplicitlyCopyable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsImplicitlyCopyablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyCopyable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyCopyable`.
"""

comptime _IsImplicitlyCopyablePredicate[T: AnyType]: Bool = conforms_to(
    T, ImplicitlyCopyable
)

comptime AllDefaultable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsDefaultablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Defaultable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Defaultable`.
"""

comptime _IsDefaultablePredicate[T: AnyType]: Bool = conforms_to(T, Defaultable)

comptime AllEquatable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsEquatablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Equatable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Equatable`.
"""

comptime _IsEquatablePredicate[T: AnyType]: Bool = conforms_to(T, Equatable)

comptime AllHashable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsHashablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `Hashable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `Hashable`.
"""

comptime _IsHashablePredicate[T: AnyType]: Bool = conforms_to(T, Hashable)

comptime AllImplicitlyDestructible[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsImplicitlyDestructiblePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `ImplicitlyDestructible`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `ImplicitlyDestructible`.
"""

comptime _IsImplicitlyDestructiblePredicate[T: AnyType]: Bool = conforms_to(
    T, ImplicitlyDestructible
)

comptime AllRegisterPassable[*Ts: AnyType]: Bool = Ts.all_satisfies[
    _IsRegisterPassablePredicate,
]()
"""Evaluates to `True` if all types in `Ts` conform to `RegisterPassable`, `False` otherwise.

Parameters:
    Ts: The types to check for conformance to `RegisterPassable`.
"""

comptime _IsRegisterPassablePredicate[T: AnyType]: Bool = conforms_to(
    T, RegisterPassable
)
