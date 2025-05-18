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
"""Implements the prelude package.  This package provide the public entities
that are automatically imported into every Mojo program.
"""

from collections import InlineArray, KeyElement, List, Optional
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
from hashlib.hash import Hashable, hash

from builtin.anytype import AnyType, UnknownDestructibility
from builtin.bool import Bool, Boolable, ImplicitlyBoolable, all, any
from builtin.breakpoint import breakpoint
from builtin.builtin_slice import Slice, slice
from builtin.comparable import (
    Comparable,
    GreaterThanComparable,
    GreaterThanOrEqualComparable,
    LessThanComparable,
    LessThanOrEqualComparable,
)
from builtin.constrained import constrained
from builtin.coroutine import AnyCoroutine, Coroutine, RaisingCoroutine
from builtin.debug_assert import (
    WRITE_MODE,
    WRITE_MODE_MEM,
    WRITE_MODE_REG,
    debug_assert,
)
from builtin.dtype import DType
from builtin.equality_comparable import EqualityComparable
from builtin.error import Error
from builtin.file import FileHandle, open
from builtin.file_descriptor import FileDescriptor
from builtin.float_literal import FloatLiteral
from builtin.floatable import Floatable, FloatableRaising
from builtin.format_int import bin, hex, oct
from builtin.identifiable import Identifiable
from builtin.int import (
    ImplicitlyIntable,
    Indexer,
    Int,
    Intable,
    IntableRaising,
    index,
)
from builtin.int_literal import IntLiteral
from builtin.io import input, print
from builtin.len import Sized, SizedRaising, UIntSized, len
from builtin.math import (
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
from builtin.rebind import rebind
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
)
from builtin.sort import partition, sort
from builtin.str import Stringable, StringableRaising
from builtin.string_literal import StringLiteral
from builtin.swap import swap
from builtin.tuple import Tuple
from builtin.type_aliases import (
    AnyTrivialRegType,
    ImmutableAnyOrigin,
    ImmutableOrigin,
    MutableAnyOrigin,
    MutableOrigin,
    Origin,
    OriginSet,
    StaticConstantOrigin,
)
from builtin.uint import UInt
from builtin.value import Copyable, Defaultable, ExplicitlyCopyable, Movable
from builtin.variadics import VariadicList, VariadicListMem, VariadicPack
from documentation import doc_private
from memory import AddressSpace, Pointer, Span

from utils import Writable, Writer
