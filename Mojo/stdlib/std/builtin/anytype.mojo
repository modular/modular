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
"""Existential-type helpers built on `AnyType`.

This module defines `Some` and `SomeTypeList`, aliases that let a function
signature express "any type conforming to trait `X`" without introducing an
explicit type parameter.
"""

from std.builtin.variadics import _MLIR
import std.traits


comptime __SomeImpl[Trait: type_of(std.traits.AnyType), T: Trait] = T
comptime __SomeTypeListImpl[
    Trait: type_of(std.traits.AnyType),
    values: _MLIR.KGENParamListType[Trait],
] = TypeList[Trait=Trait, values]()

comptime Some[Trait: type_of(std.traits.AnyType)] = __SomeImpl[Trait, ...]
"""An alias allowing users to tersely express that a function argument is an
instance of a type that implements a trait or trait composition.

For example, instead of writing

```mojo
def foo[T: Intable, //](x: T) -> Int:
    return x.__int__()
```

one can write:

```mojo
def foo(x: Some[Intable]) -> Int:
    return x.__int__()
```

Parameters:
    Trait: The trait or trait composition that the argument type must implement.
"""

comptime SomeTypeList[Trait: type_of(std.traits.AnyType)] = __SomeTypeListImpl[
    Trait, ...
]
"""An alias allowing users to tersely express that a function argument is a
list of types that implement a trait or trait composition. This is particularly
useful for variadic packs.

For example, instead of writing

```mojo
def foo[*arg_types: Copyable](*args: *arg_types) -> Int: ...
```

one can write:

```mojo
def foo(*args: *SomeTypeList[Copyable]) -> Int: ...
```

Parameters:
    Trait: The trait or trait composition that the argument types must implement.
"""
