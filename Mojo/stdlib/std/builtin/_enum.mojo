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
"""The runtime half of `__union` and `__enum`: the compiler declares their
members against this module, and nothing in it is user-facing.

For

```mojo
__union Token:
    case eof()
    case ident(String)
    case span(Int, Int)
```

the compiler generates, written as Mojo:

```mojo
struct Token(__Union):
    comptime _enum_case_names = ["eof", "ident", "span"]
    comptime _enum_case_types = [Tuple[], Tuple[String], Tuple[Int, Int]]
    var __union_storage: __UnionStorage[Tuple[], Tuple[String], Tuple[Int, Int]]

    @staticmethod
    def eof() -> Token:
        return Token.__enum_init[0]()

    @staticmethod
    def ident(var payload0: String) -> Token:
        return Token.__enum_init[1](payload0^)

    @staticmethod
    def span(var payload0: Int, var payload1: Int) -> Token:
        return Token.__enum_init[2](payload0^, payload1^)
```

`__UnionStorage` holds the active payload and tag and supplies its lifecycle.
The union's lifecycle uses ordinary struct synthesis over that field.
`__Union` supplies initialization and access defaults through `EnumLike`.
The compiler finds these declarations by module path.

An `__enum` stores only its scalar tag and conforms to `__Enum`. Its cases
take no parentheses and become `comptime` values. For `case eof`, the
compiler generates `comptime eof: Token = Token.__enum_init[0]()`.
"""

from std.builtin.enum_like import EnumLike
from std.builtin.variadics import TypeList, _MLIR
from std.sys import align_of, size_of
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)


@doc_hidden
@always_inline("builtin")
def __enum_discriminant_dtype(num_cases: Int) -> DType:
    """Returns the discriminant type for `num_cases` cases."""
    # `__EnumTag[n]` is `Scalar[__enum_discriminant_dtype(n)]`.
    # In generic code `n` is unknown, so that dtype parameter stays an
    # unevaluated call. Printed IR does not parse such a call back to the same
    # type. The parser inlines a builtin function instead, so the parameter
    # becomes a `pop.select` on `n`. A builtin body must fold, so it uses
    # `pop.select` and not `if`.
    return __mlir_op.`pop.select`(
        (num_cases <= 256).__mlir_bool__(),
        DType.uint8,
        __mlir_op.`pop.select`(
            (num_cases <= 65536).__mlir_bool__(), DType.uint16, DType.uint32
        ),
    )


comptime __EnumTag[num_cases: Int] = Scalar[
    __enum_discriminant_dtype(num_cases)
]
"""The scalar that holds the index of one of `num_cases` cases: the type of a
`__union`'s tag and of an `__enum`'s only field."""


@doc_hidden
struct __UnionStorage[case_types: _MLIR.KGENParamListType[AnyType]](
    Copyable where TypeList[Trait=AnyType, case_types].all_conforms_to[
        Copyable
    ](),
    Deinitable where TypeList[Trait=AnyType, case_types].all_conforms_to[
        Deinitable
    ](),
    ImplicitlyCopyable where TypeList[
        Trait=AnyType, case_types
    ].all_conforms_to[ImplicitlyCopyable](),
    Movable where TypeList[Trait=AnyType, case_types].all_conforms_to[
        Movable
    ](),
    RegisterPassable where TypeList[Trait=AnyType, case_types].all_conforms_to[
        RegisterPassable
    ](),
):
    """Stores a `__union`'s active payload and the index of that case.

    Cases are identified by index rather than by payload type, so several cases
    may share a payload type.

    Parameters:
        case_types: The tuple payload type of each case, `Tuple[]` for a case
            without a payload.
    """

    comptime _Cases = TypeList[Trait=AnyType, Self.case_types]

    comptime __del__is_trivial = Self._Cases.all[IsTriviallyDeinitable]()
    comptime __copy_ctor_is_trivial = Self._Cases.all[IsTriviallyCopyable]()
    comptime __move_ctor_is_trivial = Self._Cases.all[IsTriviallyMovable]()

    comptime _mlir_type = __mlir_type[
        `!kgen.untagged_union<[rebind(:`,
        type_of(Self.case_types),
        ` `,
        Self.case_types,
        `)]>`,
    ]

    # Build discriminants with `from_int=`, a plain `pop.cast`. Declaring a
    # union resolves its copy constructor, and a conversion through `SIMD.cast`
    # there would pull in `String` and the atomics, so a union could not
    # appear in the signature of anything those depend on.
    comptime _Discriminant = __EnumTag[Self._Cases.length]

    var untagged_union: Self._mlir_type
    var tag: Self._Discriminant

    @inline(.always)
    def __init__(out self, *, unsafe_uninitialized: Tuple[]):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        # Inactive bytes participate in compile-time value identity. Match
        # variant construction by clearing them before writing the payload.
        __mlir_op.`kgen.untagged_union.init`(
            Pointer(to=self.untagged_union)._mlir_value
        )

    @inline(.always)
    def __init__(
        out self, *, copy: Self
    ) where Self._Cases.all_conforms_to[Copyable]():
        comptime if Self.__copy_ctor_is_trivial:
            self.untagged_union = copy.untagged_union
            self.tag = copy.tag
            return
        self = Self(unsafe_uninitialized=())
        var discriminant = copy._discriminant()
        self._discriminant() = discriminant
        comptime for i in range(Self._Cases.length):
            # The `where` clause states the constraint; the solver does not
            # derive the element's conformance from `all_conforms_to`, so the
            # assert supplies it for the call below (as in `Tuple.__init__`).
            comptime assert conforms_to(Self._Cases()[i], Copyable)
            if discriminant == Self._Discriminant(from_int=i):
                self.payload_ptr[i]().unsafe_write(copy=copy.payload_ptr[i]()[])

    @inline(.always)
    def __init__(
        out self, *, deinit move: Self
    ) where Self._Cases.all_conforms_to[Movable]():
        comptime if Self.__move_ctor_is_trivial:
            self.untagged_union = move.untagged_union
            self.tag = move.tag
            return
        self = Self(unsafe_uninitialized=())
        var discriminant = move._discriminant()
        self._discriminant() = discriminant
        comptime for i in range(Self._Cases.length):
            comptime assert conforms_to(Self._Cases()[i], Movable)
            if discriminant == Self._Discriminant(from_int=i):
                self.payload_ptr[i]().unsafe_write_move_from(
                    move.payload_ptr[i]()
                )

    @inline(.always)
    def __deinit__(deinit self) where Self._Cases.all_conforms_to[Deinitable]():
        var discriminant = self._discriminant()
        comptime for i in range(Self._Cases.length):
            comptime assert conforms_to(Self._Cases()[i], Deinitable)
            if discriminant == Self._Discriminant(from_int=i):
                self.payload_ptr[i]().unsafe_deinit_pointee()

    @inline(.nodebug)
    def _discriminant(ref self) -> ref[self] Self._Discriminant:
        return Pointer[Self._Discriminant, origin_of(self)](
            _mlir_value=Pointer(to=self.tag)._mlir_value
        )[]

    @inline(.nodebug)
    def get_discriminant(self) -> Int:
        """Returns the index of the active case.

        Returns:
            The index of the active case.
        """
        return Int(mlir_value=self._discriminant().__mlir_index__())

    @inline(.nodebug)
    def payload_ptr[
        id: Int
    ](ref self) -> Pointer[Self._Cases()[id], origin_of(self)]:
        """Returns a pointer to the payload of case `id`.

        Only valid when case `id` is active.

        Parameters:
            id: The index of the case.

        Returns:
            A pointer to the payload of case `id`.
        """
        var member = __mlir_op.`lit.ref.union.cast`[index=id.__mlir_index__()](
            __get_mvalue_as_litref(self.untagged_union)
        )
        var typed_member = rebind[
            Pointer[
                Self._Cases()[id], origin_of(self.untagged_union)
            ]._mlir_lit_ref
        ](member)
        return Pointer[Self._Cases()[id], origin_of(self)](
            _mlir_value=__mlir_op.`lit.ref.to_pointer`(typed_member)
        )


trait __Enum(EnumLike):
    """Provides the defaults of an `__enum` stored as its discriminant."""

    comptime _EnumCases = TypeList[Trait=AnyType, Self._enum_case_types]
    """The enum's cases, as a `TypeList` of their payload types."""

    comptime _EnumStorage = __EnumTag[Self._EnumCases.length]
    """The scalar an `__enum` stores its discriminant in."""

    # The return type spells `Self._EnumStorage` out: with the alias here,
    # `type_of(result.__get_enum_storage())` in `__enum_init` is an alias
    # reference, and constructing the storage from it can no longer infer
    # `Scalar`'s `dtype` parameter.
    @inline(.always)
    def __get_enum_storage(
        ref self,
    ) -> ref[self] Scalar[__enum_discriminant_dtype(Self._EnumCases.length)]:
        comptime assert (
            size_of[Self]() == size_of[Self._EnumStorage]()
            and align_of[Self]() == align_of[Self._EnumStorage]()
        ), "an '__enum' must have its discriminant as its only field"
        return Pointer(to=self).unsafe_bitcast[Self._EnumStorage]()[]

    @staticmethod
    @inline(.always)
    def __enum_init[id: Int](out result: Self):
        comptime assert (
            0 <= id < Self._EnumCases.length
        ), "case index must be below the case count"
        comptime assert (
            Self._enum_elt_type_for_case[id] == Tuple[]
        ), "an enum case must not have a payload"
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )
        result.__get_enum_storage() = type_of(result.__get_enum_storage())(
            from_int=id
        )

    @inline(.always)
    def _get_enum_discriminant(self) -> Int:
        return Int(mlir_value=self.__get_enum_storage().__mlir_index__())

    @inline(.always)
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] Self._enum_elt_type_for_case[id]:
        # TODO(MOCO-5080): remove the empty payload reference when `EnumLike`
        # no longer requires payload access.
        comptime Payload = Self._enum_elt_type_for_case[id]
        comptime assert Payload == Tuple[]
        return Pointer(to=self).unsafe_bitcast[Payload]()[]


trait __Union(EnumLike):
    """Provides the defaults every `__union` gets.

    The compiler generates each one's only field, `__union_storage`, an
    `__UnionStorage` over its case types. `__get_enum_storage` reaches that
    field; the two `EnumLike` methods read the storage and `__enum_init`
    writes it."""

    @inline(.always)
    def __get_enum_storage(
        ref self,
    ) -> ref[self] __UnionStorage[Self._enum_case_types]:
        """Returns the storage field, the union's only field."""
        comptime Storage = __UnionStorage[Self._enum_case_types]
        comptime assert (
            size_of[Self]() == size_of[Storage]()
            and align_of[Self]() == align_of[Storage]()
        ), "a '__union' must have its storage as its only field"
        # A cast, not a field access: a register-passable union is lowered as
        # its storage itself, and a field access on the generic `Self` relies
        # on the compiler folding it away for such a union.
        return Pointer(to=self).unsafe_bitcast[Storage]()[]

    @staticmethod
    @inline(.always)
    def __enum_init[id: Int, *Ts: Movable](var *payload: *Ts, out result: Self):
        """Constructs the value of case `id` from its payload's elements. Each
        one is written straight into the storage."""
        comptime Storage = __UnionStorage[Self._enum_case_types]
        # The compiler binds `id` and `Ts` from the case it emits the call for;
        # a hand-written call is checked the same way.
        comptime assert (
            0 <= id < Storage._Cases.length
        ), "case index must be below the case count"
        comptime assert (
            Tuple[*Ts] == Storage._Cases()[id]
        ), "payload elements must build the case's payload type"
        # The payload arrives as a pack and each element is written into the
        # storage where it will live, instead of being gathered into one
        # payload value that is then moved in as a whole. That would move each
        # element twice, into the value and then with it into the storage, and
        # the second move survives for a move constructor with effects and in
        # unoptimized builds. The element types cannot be spelled from the
        # case's payload type, since a pack's types must be a variadic
        # parameter, so `Ts` is a parameter of its own, inferred from the
        # arguments the synthesized case member passes on unchanged.
        # The result is built in place: its storage is created uninitialized
        # and given the discriminant, then the payload slot is filled.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )
        var storage = Pointer(to=result.__get_enum_storage())
        __get_address_as_uninit_lvalue(storage._mlir_value) = Storage(
            unsafe_uninitialized=()
        )
        storage[]._discriminant() = Storage._Discriminant(from_int=id)
        var slot = storage[].payload_ptr[id]()

        # The assertion `Tuple[*Ts] == Storage._Cases()[id]` above makes the
        # slot's type `Tuple[*Ts]`, but the type checker cannot derive that
        # equality, so `rebind` states it and checks it once the enum is
        # instantiated.
        var elements = Pointer(to=rebind[Tuple[*Ts]](slot[]))

        def init_element[i: Int](var element: Ts[i]) {elements}:
            Pointer(to=elements[][i]).unsafe_write(element^)

        payload^.consume_elements(init_element)

    @inline(.always)
    def _get_enum_discriminant(self) -> Int:
        return self.__get_enum_storage().get_discriminant()

    @inline(.always)
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        return self.__get_enum_storage().payload_ptr[id]()[]
