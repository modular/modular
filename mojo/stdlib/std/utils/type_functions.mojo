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
"""Provides type functions for compile-time type manipulation.

Type functions are `comptime` declarations that produce a type from compile-time
parameter inputs. Unlike regular `fn` functions which accept and return runtime
values, type functions operate entirely at compile time -- they take `comptime`
parameters and evaluate to a type, with no runtime component.
"""


comptime ConditionalType[
    Trait: type_of(AnyType),
    //,
    *,
    If: Bool,
    Then: Trait,
    Else: Trait,
] = Variadic.concat_types[
    Variadic.filter_types[
        *Variadic.types[T=Trait, Then],
        predicate = _ConditionalTypePredicate[Trait, If],
    ],
    Variadic.types[Else],
][
    0
]
"""A type function that conditionally selects between two types.

This type function evaluates a compile-time boolean condition and produces
either type `Then` (if the condition is True) or type `Else` (if the condition
is False). It is the type-level equivalent of the ternary conditional expression
`Then if If else Else`.

Parameters:
    Trait: A trait that both `Then` and `Else` must conform to.
    If: A compile-time boolean that determines which type to select.
    Then: The type to produce if the condition is True.
    Else: The type to produce if the condition is False.

Returns:
    Type `Then` if `If` is True, otherwise type `Else`.

Examples:
    ```mojo
    from utils.type_functions import ConditionalType
    from sys import size_of

    struct Wrapper[T: AnyType]:
        comptime StorageType = ConditionalType[
            Trait=ImplicitlyDestructible,
            If=size_of[Self.T]() > 0,
            Then=List[Byte],
            Else=NoneType,
        ]

        var storage: Self.StorageType
    ```
"""

comptime _ConditionalTypePredicate[
    Trait: type_of(AnyType),
    condition: Bool,
    T: Trait,
] = condition
