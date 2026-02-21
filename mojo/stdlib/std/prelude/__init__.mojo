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
"""Standard library prelude: fundamental types, traits, and operations auto-imported.

This package's contents form the basic vocabulary of Mojo programming that every
developer uses. It is implicitly imported to every Mojo program.

The `prelude` package contains the core types, traits, and functions that are
automatically imported into every Mojo program. It provides the foundational
building blocks of the language including basic types (Int, String, Bool),
essential traits (Copyable, Movable, Equatable), memory primitives (Pointer,
Span), and common operations (print, len, range). This package defines the
default namespace that makes Mojo code immediately usable without explicit
imports.
"""

from collections import Dict, InlineArray, KeyElement, List, Optional
from collections.string import (
    Codepoint,
    StaticString,
    String,
    StringSlice,
    ascii,
    atof,
    atol,
    chr,
    ord,
)
from format import Writable, Writer
from hashlib.hash import Hashable, hash
from io.file import FileHandle, open
from io.file_descriptor import FileDescriptor
from io.io import input, print

from builtin.anytype import (
    AnyType,
    Some,
    UnknownDestructibility,
    ImplicitlyDestructible,
)
from builtin.bool import Bool, Boolable, all, any
from builtin.breakpoint import breakpoint
from builtin.builtin_slice import Slice, slice
from builtin.comparable import Comparable, Equatable
from builtin.constrained import constrained
from builtin.coroutine import AnyCoroutine, Coroutine, RaisingCoroutine
from builtin.debug_assert import debug_assert
from builtin.dtype import DType
from builtin.error import Error
from builtin.float_literal import FloatLiteral
from builtin.floatable import Floatable, FloatableRaising
from builtin.format_int import bin, hex, oct
from builtin.identifiable import Identifiable
from builtin.int import (
    Indexer,
    Int,
    Intable,
    IntableRaising,
    index,
)
from builtin.int_literal import IntLiteral
from builtin.len import Sized, SizedRaising, UIntSized, len
from math.math import (
    Absable,
    Powable,
    Roundable,
    abs,
    divmod,
    max,
    min,
    pow,
    round,
)
from builtin.none import NoneType
from builtin.range import range
from builtin.rebind import (
    rebind,
    rebind_var,
    trait_downcast,
    trait_downcast_var,
)
from builtin.repr import Representable, repr
from builtin.reversed import ReversibleRange, reversed
from builtin.simd import (
    SIMD,
    BFloat16,
    Byte,
    Float8_e4m3fn,
    Float8_e4m3fnuz,
    Float8_e5m2,
    Float8_e5m2fnuz,
    Float8_e8m0fnu,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Int256,
    Scalar,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    UInt256,
    UInt,
)
from builtin.sort import partition, sort
from builtin.str import Stringable
from builtin.string_literal import StringLiteral
from builtin.swap import swap
from builtin.tuple import Tuple
from builtin.type_aliases import (
    __TypeOfAllTypes,
    AnyOrigin,
    ImmutAnyOrigin,
    ImmutOrigin,
    MutAnyOrigin,
    MutOrigin,
    Never,
    Origin,
    OriginSet,
    StaticConstantOrigin,
    ExternalOrigin,
    ImmutExternalOrigin,
    MutExternalOrigin,
)
from builtin.value import (
    Copyable,
    Defaultable,
    ExplicitlyCopyable,
    ImplicitlyCopyable,
    Movable,
    materialize,
    RegisterPassable,
    TrivialRegisterPassable,
)
from builtin.variadics import (
    Variadic,
    VariadicList,
    VariadicListMem,
    VariadicPack,
)
from documentation import doc_private
from iter import (
    Iterable,
    Iterator,
    StopIteration,
    enumerate,
    iter,
    map,
    next,
    zip,
)
from memory import (
    alloc,
    AddressSpace,
    LegacyOpaquePointer,
    LegacyUnsafePointer,
    ImmutOpaquePointer,
    MutOpaquePointer,
    OpaquePointer,
    Pointer,
    Span,
    ImmutUnsafePointer,
    MutUnsafePointer,
    UnsafePointer,
)
