# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias Index = __mlir_type.index
alias string = __mlir_type.`!kgen.string`
alias float = __mlir_type.`!pop.scalar<f64>`

alias AnyTrivialRegType = __mlir_type.`!kgen.type`
alias ImmutableOrigin = __mlir_type.`!lit.origin<0>`
alias MutableOrigin = __mlir_type.`!lit.origin<1>`
alias ImmutableAnyOrigin = __mlir_attr.`#lit.any.origin : !lit.origin<0>`
alias MutableAnyOrigin = __mlir_attr.`#lit.any.origin<1>: !lit.origin<1>`
alias OriginSet = __mlir_type.`!lit.origin.set`


alias `0` = __mlir_attr.`0 : index`
alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`
alias `3` = __mlir_attr.`3 : index`
alias `4` = __mlir_attr.`4 : index`
alias `5` = __mlir_attr.`5 : index`
alias `6` = __mlir_attr.`6 : index`
alias `7` = __mlir_attr.`7 : index`
alias `8` = __mlir_attr.`8 : index`
alias `9` = __mlir_attr.`9 : index`
alias `10` = __mlir_attr.`10 : index`
alias `42` = __mlir_attr.`42 : index`
alias `123` = __mlir_attr.`123 : index`
alias `True` = __mlir_attr.`1 : i1`
alias `False` = __mlir_attr.`0 : i1`


@value
@register_passable("trivial")
struct Origin[mut: Bool]:
    alias _mlir_type = __mlir_type[
        `!lit.origin<`,
        mut.value,
        `>`,
    ]

    var _mlir_origin: Self._mlir_type

    alias cast_from = _lit_mut_cast[result_mutable=mut]

    @always_inline("builtin")
    @implicit
    fn __init__(out self, mlir_origin: Self._mlir_type):
        """Initialize an Origin from a raw MLIR `!lit.origin` value.

        Args:
            mlir_origin: The raw MLIR origin value."""
        self._mlir_origin = mlir_origin


struct _lit_mut_cast[
    mut: Bool, //,
    result_mutable: Bool,
    operand: Origin[mut],
]:
    alias result = __mlir_attr[
        `#lit.origin.mutcast<`,
        operand._mlir_origin,
        `> : !lit.origin<`,
        result_mutable.value,
        `>`,
    ]


# Static constants are a named subset of the global origin.
alias StaticConstantOrigin = __mlir_attr[
    `#lit.origin.field<`,
    `#lit.static.origin : !lit.origin<0>`,
    `, "__constants__"> : !lit.origin<0>`,
]


struct _lit_indirect_origin[mut: Bool, //, base: Origin[mut]._mlir_type]:
    alias result = __mlir_attr[
        `#lit.indirect.origin<`,
        Self.base,
        `> : `,
        __type_of(Self.base),
    ]


# ===----------------------------------------------------------------------=== #
# Builtin Types
# ===----------------------------------------------------------------------=== #


trait KeyElement(Copyable, Movable):
    pass


@register_passable
struct Error:
    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, value: StringLiteral):
        pass

    fn __del__(owned self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    # A method for testing.
    fn use(self):
        pass


@register_passable("trivial")
struct NoneType:
    alias _mlir_type = __mlir_type.`!kgen.none`
    """Raw MLIR type of the `None` value."""

    var _value: Self._mlir_type

    # FIXME: Fix representation of None literal to remove this.
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.`!kgen.none`):
        self._value = value


@value
@nonmaterializable(Int)
@register_passable("trivial")
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    alias _zero = IntLiteral[
        __mlir_attr.`#pop.int_literal<0> : !pop.int_literal`
    ]()
    alias _one = IntLiteral[
        __mlir_attr.`#pop.int_literal<1> : !pop.int_literal`
    ]()

    @always_inline("builtin")
    fn __init__(out self):
        """Constructor for any value."""
        pass

    @always_inline("builtin")
    fn __ne__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<ne `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return self != Self._zero

    @always_inline("builtin")
    fn __mul__(
        self,
        rhs: IntLiteral[_],
        out result: IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<mul `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ],
    ):
        result = __type_of(result)()


@value
@nonmaterializable(FloatDyn)
@register_passable("trivial")
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`]:
    @always_inline("builtin")
    fn __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    fn __init__(
        value: IntLiteral[_],
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<int_to_float_literal<`,
                value.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = __type_of(result)()


@value
@register_passable("trivial")
struct FloatDyn:
    var value: __mlir_type.`!pop.scalar<f64>`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<f64>`):
        self.value = value

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: FloatLiteral):
        self = __mlir_attr[
            `#pop<float_literal_convert<`, +value.value, `>> : !pop.scalar<f64>`
        ]

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral):
        self = FloatLiteral(value)


@value
@register_passable("trivial")
struct Int(AnyRPTrivialType, Copyable):
    var value: Index

    @always_inline("builtin")
    fn __init__(out self):
        self.value = __mlir_op.`index.constant`[value = __mlir_attr.`0:index`]()

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Index):
        self.value = value

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral[_]):
        self.value = __mlir_attr[
            `#pop<int_literal_convert<`, +value.value, `, 0>> : index`
        ]

    @always_inline("builtin")
    fn __add__(lhs, rhs: Int) -> Int:
        return __mlir_op.`index.add`(lhs.value, rhs.value)

    @always_inline("builtin")
    fn __sub__(lhs, rhs: Int) -> Int:
        return __mlir_op.`index.sub`(lhs.value, rhs.value)

    @always_inline("builtin")
    fn __mul__(lhs, rhs: Int) -> Int:
        return __mlir_op.`index.mul`(lhs.value, rhs.value)

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Int):
        self = self + rhs

    @always_inline("builtin")
    fn __eq__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](lhs.value, rhs.value)

    @always_inline("builtin")
    fn __lt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](rhs.value, lhs.value)

    @always_inline("builtin")
    fn __gt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](lhs.value, rhs.value)

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return not (self == 0)

    @always_inline("builtin")
    fn __index__(self) -> __mlir_type.index:
        return self.value


@value
@register_passable("trivial")
struct UInt8:
    fn __init__(out self):
        pass


alias Byte = UInt8


@value
@register_passable("trivial")
struct Span[
    mut: Bool, //,
    T: Copyable & Movable,
    origin: Origin[mut],
]:
    # Field
    var _data: UnsafePointer[T, mut=mut, origin=origin]
    var _len: Int

    fn __init__(out self):
        self._data = UnsafePointer[T, mut=mut, origin=origin]()
        self._len = 0

    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[T, mut=mut, origin=origin,]:
        return self._data


@value
@register_passable("trivial")
struct StringLiteral[value: __mlir_type.`!kgen.string`]:
    @always_inline("builtin")
    fn __init__(out self):
        pass

    @always_inline("nodebug")
    fn __eq__(self, other: StringLiteral) -> Bool:
        return Bool()

    # TODO(MSTDL-1327): Reduce pain when string literals can't be
    # non-materializable by making them merge into StaticString.  They should
    # eventually merge into String through nonmaterialization.
    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(StringLiteral[_]),
    ](self) -> StaticString:
        return self


@register_passable("trivial")
struct StringSlice[mut: Bool, //, origin: Origin[mut]]:
    var _slice: Span[Byte, origin]

    @implicit
    fn __init__[
        origin: ImmutableOrigin, //
    ](out self: StringSlice[origin], ref [origin]value: String):
        self._slice = Span[Byte, origin]()

    @implicit
    fn __init__(out self: StaticString, lit: StringLiteral):
        pass

    @always_inline
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[Byte, mut=mut, origin=origin]:
        return self._slice.unsafe_ptr()

    @always_inline
    fn byte_length(self) -> Int:
        return self._slice._len


alias StaticString = StringSlice[StaticConstantOrigin]


@always_inline("builtin")
fn _get_kgen_string[
    string: StaticString, extra: VariadicList[StaticString]
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.value,
        `> : !kgen.string`,
    ]


@always_inline("builtin")
fn _get_kgen_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return _get_kgen_string[string, extra]()


@always_inline("nodebug")
fn get_static_string[
    string: StaticString, *extra: StaticString
]() -> StaticString:
    return StringLiteral(_get_kgen_string[string, extra]())


struct String(KeyElement):
    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, literal: StringLiteral):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, owned existing: String):
        pass

    fn __del__(owned self):
        pass

    fn __len__(self) -> Int:
        return 0

    fn __contains__(self, substr: StringSlice[mut=False]) -> Bool:
        return True

    fn __iadd__(mut self, rhs: StringSlice[mut=False]):
        pass

    fn byte_length(self) -> Int:
        return 0

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        return UnsafePointer[UInt8]()


@value
@register_passable("trivial")
struct Bool(AnyRPTrivialType):
    var value: __mlir_type.i1

    @always_inline("builtin")
    fn __init__(out self):
        self.value = __mlir_attr.`0 : i1`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.i1):
        self.value = value

    @always_inline("builtin")
    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self.value

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return self

    @always_inline("builtin")
    fn __invert__(self) -> Bool:
        return self  # Incorrect impl

    @always_inline("builtin")
    fn __and__(self, rhs: Bool) -> Bool:
        return __mlir_op.`pop.and`(self.value, rhs.value)


@register_passable("trivial")
struct Slice:
    @implicit
    fn __init__(out self, end: Index):
        pass

    fn __init__(out self, start: Index, end: Index):
        return

    fn __init__[
        T0: AnyTrivialRegType, T1: AnyTrivialRegType, T2: AnyTrivialRegType
    ](out self, start: T0, end: T1, step: T2):
        pass


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


# A linear type, see
# https://www.notion.so/modularai/Linear-Types-14a1044d37bb809ab074c990fe1a84e3.
trait UnknownDestructibility:
    pass


@explicit_destroy
trait ExplicitlyDestroyedCopyable:
    fn __copyinit__(out self, existing: Self, /):
        pass


trait Copyable:
    fn __copyinit__(out self, existing: Self, /):
        pass


@explicit_destroy
trait ExplicitlyDestroyedMovable:
    fn __moveinit__(out self, owned existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(out self, owned existing: Self, /):
        pass


trait ExplicitlyCopyable:
    fn copy(self) -> Self:
        ...


trait AnyType:
    fn __del__(owned self, /):
        ...


alias ImplicitlyDestructible = AnyType


@register_passable("trivial")
trait AnyRPTrivialType:
    pass


# ===----------------------------------------------------------------------=== #
# Builtin Collection Stubs
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct VariadicList[type: AnyTrivialRegType]:
    alias _mlir_type = __mlir_type[`!kgen.variadic<`, type, `>`]

    var value: Self._mlir_type

    @implicit
    fn __init__(out self, *value: type):
        self = value

    @implicit
    fn __init__(out self, value: Self._mlir_type):
        self.value = value


# Helper to compute the union of two origins:
# TODO: parametric aliases would be nice.
struct _lit_origin_union[
    mut: Bool, //,
    a: Origin[mut].type,
    b: Origin[mut].type,
]:
    alias result = __mlir_attr[
        `#lit.origin.union<`,
        a,
        `,`,
        b,
        `> : !lit.origin<`,
        mut.value,
        `>`,
    ]


@value
struct _VariadicListMemIter[
    elt_is_mutable: Bool, //,
    elt_type: AnyType,
    elt_origin: Origin[elt_is_mutable],
    list_origin: ImmutableOrigin,
    is_owned: Bool,
]:
    """Iterator for VariadicListMem.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicListMem.
    """

    alias variadic_list_type = VariadicListMem[elt_type, elt_origin, is_owned]

    var index: Int
    var src: Pointer[Self.variadic_list_type, list_origin]

    fn __next__(mut self) -> Self.variadic_list_type.reference_type:
        while True:
            pass

    fn __has_next__(self) -> Bool:
        return False


struct VariadicListMem[
    elt_is_mutable: Bool, //,
    element_type: AnyType,
    origin: Origin[elt_is_mutable],
    is_owned: Bool,
]:
    alias reference_type = Pointer[element_type, origin]
    alias _mlir_ref_type = Self.reference_type._mlir_type
    alias _mlir_type = __mlir_type[
        `!kgen.variadic<`, Self._mlir_ref_type, `, read_mem>`
    ]

    @implicit
    fn __init__(
        out self,
        value: Self._mlir_type,
    ):
        pass

    @implicit
    fn __init__(
        out self,
        value: __mlir_type[`!kgen.variadic<`, Self._mlir_ref_type, `, mut>`],
    ):
        pass

    @implicit
    fn __init__(
        out self,
        value: __mlir_type[
            `!kgen.variadic<`, Self._mlir_ref_type, `, owned_in_mem>`
        ],
    ):
        pass

    fn __getitem__(
        self, idx: Int
    ) -> ref [
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        Origin[elt_is_mutable]
        .cast_from[__origin_of(origin, self)]
        .result
    ] element_type:
        while True:
            pass

    fn __iter__(
        self,
        out result: _VariadicListMemIter[
            element_type, origin, __origin_of(self), is_owned
        ],
    ):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return __type_of(result)(0, Pointer(to=self))


alias _AnyTypeMetaType = __type_of(AnyType)


@register_passable
struct VariadicPack[
    elt_is_mutable: Bool, //,
    is_owned: Bool,
    origin: Origin[elt_is_mutable],
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
]:
    alias _mlir_pack_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        origin._mlir_origin,
        `>`,
    ]

    fn __init__(out self, value: Self._mlir_pack_type):
        pass

    fn __getitem__[
        index: Int
    ](self) -> ref [Self.origin] element_types[index.value]:
        while True:
            pass


@register_passable
struct __ParameterClosureCaptureList[
    fn_type: AnyTrivialRegType, fn_ref: fn_type
]:
    alias type = __mlir_type.`!kgen.pointer<none>`
    var value: Self.type

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Self.type):
        self.value = value

    # Parameter closure invariant requires this function be marked 'capturing'.
    @parameter
    @always_inline
    fn __init__(out self):
        self.value = __mlir_op.`kgen.capture_list.create`[callee=fn_ref]()

    @always_inline
    fn __copyinit__(out self, existing: Self):
        self.value = __mlir_op.`kgen.capture_list.copy`[callee=fn_ref](
            existing.value
        )

    @always_inline
    fn __del__(owned self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    fn expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)


@value
@register_passable("trivial")
struct AddressSpace:
    """Address space of the pointer."""

    var _value: Int

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    alias GENERIC = AddressSpace(0)

    @always_inline("builtin")
    fn __index__(self) -> __mlir_type.index:
        return self._value.value


@value
@register_passable("trivial")
struct Pointer[
    mut: Bool, //,
    type: AnyType,
    origin: Origin[mut],
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        origin._mlir_origin,
        `, `,
        address_space._value.value,
        `>`,
    ]

    var _value: Self._mlir_type

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, _mlir_value: Self._mlir_type):
        self._value = _mlir_value

    @always_inline("nodebug")
    fn __init__(out self, *, ref [origin, address_space._value.value]to: type):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(_mlir_value=__get_mvalue_as_litref(to))

    @staticmethod
    @always_inline("nodebug")
    fn address_of(ref [origin, address_space]value: type) -> Self:
        return Pointer(_mlir_value=__get_mvalue_as_litref(value))

    fn __getitem__(self) -> ref [origin, address_space] type:
        return __get_litref_as_mvalue(self._value)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Pointer[type, _, address_space]) -> Bool:
        return True

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: __type_of(Pointer[type, _, address_space]),
    ](self) -> Pointer[
        mut = mut & other_type.origin.mut,
        type=type,
        origin = __origin_of(origin, other_type.origin),
        address_space=address_space,
    ]:
        return self._value  # allow lit.ref to convert.


struct Tuple[*element_types: AnyType]:
    @implicit
    fn __init__(out self, *args: *element_types):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, owned existing: Self):
        pass

    fn __getitem__[i: Int](ref self) -> ref [self] element_types[i.value]:
        while __mlir_attr.true:
            pass


@register_passable("trivial")
struct UnsafePointer[
    T: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`, T, `,`, address_space._value.value, `>`
    ]
    var address: Self._mlir_type

    fn __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @implicit
    @always_inline("builtin")
    fn __init__(out self, value: Self._mlir_type):
        self.address = value

    @staticmethod
    fn address_of(ref [address_space]arg: T) -> Self:
        return Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(arg)))

    fn __getitem__(
        self,
    ) -> ref [Self.origin, address_space] T:
        while __mlir_attr.true:
            pass

    fn __getitem__(self, offset: Int) -> ref [Self.origin, address_space] T:
        while __mlir_attr.true:
            pass

    # This returns a reference to an element with an origin specified by as a
    # unique reference from this pointer.  The returned reference is always
    # mutable.
    fn get_unique_item_ref[
        self_origin: ImmutableOrigin
    ](ref [self_origin]self, offset: Int = 0) -> ref [
        Origin[True].cast_from[_lit_indirect_origin[self_origin].result].result,
        address_space,
    ] T:
        while __mlir_attr.true:
            pass


@value
@register_passable("trivial")
struct _StridedRangeIterator:
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    fn __next__(mut self) -> Int:
        var result = self.start
        self.start += self.step
        return result


# ===-----------------------------------------------------------------------===#
# parameter_for
# ===-----------------------------------------------------------------------===#


trait _IntNext(Copyable):
    fn __next__(mut self) -> Int:
        ...


trait _IntIter(_IntNext):
    fn __has_next__(self) -> Bool:
        ...


trait _IntIterable(_IntIter):
    fn __iter__(self) -> Self:
        ...


trait _StridedIterable(_IntIter):
    fn __iter__(self) -> _StridedRangeIterator:
        ...


struct _ParamForIterator[IteratorT: Copyable]:
    var next_it: IteratorT
    var value: Int
    var stop: Bool

    fn __init__(out self, next_it: IteratorT, value: Int, stop: Bool):
        self.next_it = next_it
        self.value = value
        self.stop = stop


fn declval[T: AnyType]() -> T:
    while True:
        pass


fn parameter_for_generator[
    T: _IntIterable,
](range: T) -> _ParamForIterator[__type_of(declval[T]().__iter__())]:
    return _generator(range.__iter__())


fn parameter_for_generator[
    T: _StridedIterable,
](range: T) -> _ParamForIterator[__type_of(declval[T]().__iter__())]:
    return _generator(range.__iter__())


fn _generator[
    IteratorT: _IntIter
](it: IteratorT, out result: _ParamForIterator[IteratorT]):
    if it.__has_next__():
        var next_it = it
        var value = next_it.__next__()
        return _ParamForIterator(next_it, value, False)
    var value: IteratorT
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(value))
    return _ParamForIterator(value^, 0, True)


struct Optional[T: Copyable & Movable]:
    fn __del__(owned self):
        pass

    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, owned value: T):
        pass

    @implicit
    fn __init__(out self, value: NoneType):
        pass

    # FIXME: None literal should be of NoneType not !kgen.none.
    @implicit
    fn __init__(out self, x: __mlir_type.`!kgen.none`):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __moveinit__(out self, owned other: Self):
        pass

    fn value(ref self) -> ref [self] T:
        while True:
            pass
