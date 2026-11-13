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
"""The runtime half of `__enum`: the compiler declares an enum's members
against this module, and nothing in it is user-facing.

For

```mojo
__enum Token:
    case eof
    case ident(String)
    case span(Int, Int)
```

the compiler generates, written as Mojo:

```mojo
struct Token(__Enum):
    comptime _enum_case_names = ["eof", "ident", "span"]
    comptime _enum_case_types = [Tuple[], Tuple[String], Tuple[Int, Int]]
    var __enum_storage: __EnumStorage[Tuple[], Tuple[String], Tuple[Int, Int]]

    comptime eof: Token = Token.__enum_init[0]()

    @staticmethod
    def ident(var payload: String) -> Token:
        return Token.__enum_init[1](payload^)

    @staticmethod
    def span(var payload0: Int, var payload1: Int) -> Token:
        return Token.__enum_init[2](payload0^, payload1^)
```

`__EnumStorage` holds the active case's payload and its index, and copies,
moves and destroys the payload; the enum's own copy, move and destructor are
the ordinary struct ones over that single field. `__Enum`, a trait refining
`EnumLike`, supplies everything else as defaults: `__get_enum_storage` reaches
the field, `__enum_init` builds the value of a case, and the two `EnumLike`
methods read the field. The compiler finds these declarations by module path.
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
def __enum_discriminant_dtype(num_cases: Int) -> DType:
    """Returns the `!kgen.variant` discriminant type for `num_cases` cases."""
    if num_cases <= 256:
        return DType.uint8
    if num_cases <= 65536:
        return DType.uint16
    return DType.uint32


@doc_hidden
struct __EnumStorage[case_types: _MLIR.KGENParamListType[AnyType]](
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
    """The storage of an `__enum`: the payload of its active case and the index
    of that case.

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
        `!kgen.variant<[rebind(:`,
        type_of(Self.case_types),
        ` `,
        Self.case_types,
        `)]>`,
    ]
    var _impl: Self._mlir_type

    # Build discriminants with `from_int=`, a plain `pop.cast`. Declaring an
    # enum resolves its copy constructor, and a conversion through `SIMD.cast`
    # there would pull in `String` and the atomics, so an enum could not
    # appear in the signature of anything those depend on.
    comptime _Discriminant = Scalar[
        __enum_discriminant_dtype(Self._Cases.length)
    ]

    @inline(.always)
    def __init__(out self, *, unsafe_uninitialized: Tuple[]):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @inline(.always)
    def __init__(
        out self, *, copy: Self
    ) where Self._Cases.all_conforms_to[Copyable]():
        comptime if Self.__copy_ctor_is_trivial:
            self._impl = copy._impl
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
            self._impl = move._impl
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
        var discr_ptr = __mlir_op.`pop.variant.discr_gep`[
            _type=__mlir_type[
                `!kgen.pointer<scalar<`,
                Self._Discriminant.dtype._mlir_value,
                `>>`,
            ]
        ](Pointer(to=self._impl)._get_kgen_pointer())
        return Pointer[_, origin_of(self)](
            _mlir_value=discr_ptr
        ).unsafe_bitcast[Self._Discriminant]()[]

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
        return {
            _mlir_value = __mlir_op.`pop.variant.bitcast`[
                _type=Pointer[Self._Cases()[id], origin_of(self)]._mlir_type,
                index=id.__mlir_index__(),
            ](Pointer(to=self._impl)._get_kgen_pointer())
        }


trait __Enum(EnumLike):
    """An `__enum`: a struct whose only field, `__enum_storage`, is an
    `__EnumStorage` over its case types. `__get_enum_storage` reaches that
    field; the two `EnumLike` methods read it and `__enum_init` writes it."""

    @inline(.always)
    def __get_enum_storage(
        ref self,
    ) -> ref[self] __EnumStorage[Self._enum_case_types]:
        """Returns the storage field, the enum's only field."""
        comptime Storage = __EnumStorage[Self._enum_case_types]
        comptime assert (
            size_of[Self]() == size_of[Storage]()
            and align_of[Self]() == align_of[Storage]()
        ), "an `__Enum` must have its storage as its only field"
        # A cast, not a field access: a register-passable enum is lowered as
        # its storage itself, and a field access on the generic `Self` relies
        # on the compiler folding it away for such an enum.
        return Pointer(to=self).unsafe_bitcast[Storage]()[]

    @staticmethod
    @inline(.always)
    def __enum_init[id: Int, *Ts: Movable](var *payload: *Ts, out result: Self):
        """Constructs the value of case `id` from its payload's elements. Each
        one is written straight into the storage."""
        comptime Storage = __EnumStorage[Self._enum_case_types]
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
