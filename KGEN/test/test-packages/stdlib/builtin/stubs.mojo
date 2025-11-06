# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias Index = __mlir_type.index
alias string = __mlir_type.`!kgen.string`
alias float = __mlir_type.`!pop.scalar<f64>`

alias AnyTrivialRegType = __mlir_type.`!kgen.type`
alias ImmutOrigin = __mlir_type.`!lit.origin<0>`
alias MutOrigin = __mlir_type.`!lit.origin<1>`
alias ImmutAnyOrigin = __mlir_attr.`#lit.any.origin : !lit.origin<0>`
alias MutAnyOrigin = __mlir_attr.`#lit.any.origin<1>: !lit.origin<1>`
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


@register_passable("trivial")
struct Origin[mut: Bool]:
    alias _mlir_type = __mlir_type[
        `!lit.origin<`,
        mut._mlir_value,
        `>`,
    ]

    var _mlir_origin: Self._mlir_type

    alias cast_from[o: Origin] = __mlir_attr[
        `#lit.origin.mutcast<`,
        o._mlir_origin,
        `> : !lit.origin<`,
        mut._mlir_value,
        `>`,
    ]

    @always_inline("builtin")
    @implicit
    fn __init__(out self, mlir_origin: Self._mlir_type):
        """Initialize an Origin from a raw MLIR `!lit.origin` value.

        Args:
            mlir_origin: The raw MLIR origin value."""
        self._mlir_origin = mlir_origin


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
        type_of(Self.base),
    ]


# ===----------------------------------------------------------------------=== #
# Builtin Types
# ===----------------------------------------------------------------------=== #


alias KeyElement = Copyable & Movable


@register_passable
struct Error(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, value: StringLiteral):
        pass

    fn __del__(deinit self):
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
    fn __le__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<le `,
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
        result = type_of(result)()

    @always_inline("builtin")
    fn __sub__(
        self,
        rhs: IntLiteral[_],
        out result: IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<sub `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    fn __neg__(self) -> type_of(0 - self):
        return 0 - self

    @always_inline("builtin")
    fn __pos__(self) -> Self:
        return self

    @always_inline("builtin")
    fn __floordiv__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<floordiv `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}


@nonmaterializable(FloatDyn)
@register_passable("trivial")
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`]:
    @always_inline("builtin")
    fn __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    fn __init__(
        val: IntLiteral[_],
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<int_to_float_literal<`,
                val.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    fn __neg__(self, out result: type_of(self * -1)):
        result = type_of(result)()

    @always_inline("builtin")
    fn __mul__(
        self,
        rhs: FloatLiteral,
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<float_literal_bin<mul `,
                value,
                `,`,
                rhs.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    fn __truediv__(
        self,
        rhs: FloatLiteral,
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<float_literal_bin<truediv `,
                value,
                `,`,
                rhs.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()


@register_passable("trivial")
struct FloatDyn:
    var _mlir_value: __mlir_type.`!pop.scalar<f64>`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<f64>`):
        self._mlir_value = value

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


@register_passable("trivial")
struct Int(AnyRPTrivialType, ImplicitlyCopyable, Intable):
    var _mlir_value: Index

    @always_inline("builtin")
    fn __init__(out self):
        self._mlir_value = __mlir_op.`index.constant`[
            value = __mlir_attr.`0:index`
        ]()

    @always_inline("builtin")
    fn __init__(out self, *, mlir_value: Index):
        self._mlir_value = mlir_value

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral[_]):
        self._mlir_value = __mlir_attr[
            `#pop<int_literal_convert<`, +value.value, `, 0>> : index`
        ]

    @always_inline("builtin")
    fn __add__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.add`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __sub__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.sub`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __mul__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.mul`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Int):
        self = self + rhs

    @always_inline("builtin")
    fn __eq__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __ne__(self, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __lt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](rhs._mlir_value, lhs._mlir_value)

    @always_inline("builtin")
    fn __gt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return not (self == 0)

    @always_inline("builtin")
    fn __mlir_index__(self) -> __mlir_type.index:
        return self._mlir_value

    @always_inline("builtin")
    fn _positive_div(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.divs`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    fn __int__(self) -> Int:
        return self

    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        pass

    @always_inline("builtin")
    fn __neg__(self) -> Int:
        return self * -1

    @always_inline("builtin")
    fn __and__(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.and`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Int):
        self = self * rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Int):
        self = self >> rhs

    @always_inline("builtin")
    fn __rshift__(self, rhs: Int) -> Int:
        pass


@register_passable("trivial")
struct UInt8:
    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, value: IntLiteral):
        pass


alias Byte = UInt8


@register_passable("trivial")
struct Span[
    mut: Bool, //,
    T: ImplicitlyCopyable & Movable,
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
        other_type: type_of(StringLiteral[_]),
    ](self) -> StaticString:
        return self


@register_passable("trivial")
struct StringSlice[mut: Bool, //, origin: Origin[mut]]:
    var _slice: Span[Byte, origin]

    @implicit
    fn __init__[
        _origin: ImmutOrigin, //
    ](out self: StringSlice[_origin], ref [_origin]value: String):
        self._slice = Span[Byte, _origin]()

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


struct String(ImplicitlyCopyable, KeyElement):
    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, literal: StringLiteral):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, deinit existing: String):
        pass

    fn __del__(deinit self):
        pass

    fn __len__(self) -> Int:
        return 0

    fn __contains__(self, substr: StringSlice[mut=False]) -> Bool:
        return True

    fn __iadd__(mut self, rhs: StringSlice[mut=False]):
        pass

    fn byte_length(self) -> Int:
        return 0

    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[UInt8, mut=False, origin = origin_of(self)]:
        return {}


@register_passable("trivial")
struct Bool(AnyRPTrivialType):
    var _mlir_value: __mlir_type.i1

    @always_inline("builtin")
    fn __init__(out self):
        self._mlir_value = __mlir_attr.`0 : i1`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.i1):
        self._mlir_value = value

    @always_inline("builtin")
    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self._mlir_value

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        return self

    @always_inline("builtin")
    fn __invert__(self) -> Bool:
        return __mlir_op.`pop.xor`(self._mlir_value, __mlir_attr.true)

    @always_inline("builtin")
    fn __and__(self, rhs: Bool) -> Bool:
        return __mlir_op.`pop.and`(self._mlir_value, rhs._mlir_value)


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


struct List[T: AnyType](Copyable, Movable):
    fn __init__(out self, *elements: T, __list_literal__: () = ()):
        pass

    fn append(mut self, var value: T):
        pass

    fn __getitem__(ref self, idx: Int) -> ref [self] T:
        pass


struct Set[T: AnyType]:
    fn __init__(out self, *elements: T, __set_literal__: () = ()):
        pass

    fn add(mut self, var value: T):
        pass


struct Dict[K: AnyType, V: ImplicitlyCopyable & Movable]:
    fn __init__(out self):
        pass

    fn __init__(
        out self,
        var keys: List[K],
        var values: List[V],
        __dict_literal__: (),
    ):
        pass

    fn __setitem__(mut self, key: K, value: V):
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
        ...


trait Copyable:
    fn __copyinit__(out self, existing: Self, /):
        ...

    fn copy(self) -> Self:
        return Self.__copyinit__(self)

    alias __copyinit__is_trivial: Bool


trait ImplicitlyCopyable(Copyable):
    pass


fn materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile time parameter into a runtime value."""
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


@explicit_destroy
trait ExplicitlyDestroyedMovable:
    fn __moveinit__(out self, deinit existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(out self, deinit existing: Self, /):
        pass

    alias __moveinit__is_trivial: Bool


trait AnyType:
    fn __del__(deinit self, /):
        ...

    alias __del__is_trivial: Bool


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
        mut._mlir_value,
        `>`,
    ]


@fieldwise_init
struct _VariadicListMemIter[
    elt_is_mutable: Bool, //,
    elt_type: AnyType,
    elt_origin: Origin[elt_is_mutable],
    list_origin: ImmutOrigin,
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

    fn __next_ref__(mut self) -> ref [elt_origin] elt_type:
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
    alias _mlir_type = __mlir_type[`!kgen.variadic<`, Self._mlir_ref_type, `>`]

    @implicit
    fn __init__(
        out self,
        value: Self._mlir_type,
    ):
        pass

    fn __getitem__(
        self, idx: Int
    ) -> ref [
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        Origin[elt_is_mutable].cast_from[origin_of(origin, self)]
    ] element_type:
        while True:
            pass

    fn __iter__(
        self,
        out result: _VariadicListMemIter[
            element_type, origin, origin_of(self), is_owned
        ],
    ):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return type_of(result)(0, Pointer(to=self))


alias _AnyTypeMetaType = type_of(AnyType)


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

    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_disable_nested_origin_exclusivity
    fn __init__(out self, value: Self._mlir_pack_type):
        pass

    fn __getitem__[index: Int](self) -> ref [Self.origin] element_types[index]:
        while True:
            pass


@register_passable
struct __ParameterClosureCaptureList[
    fn_type: AnyTrivialRegType, fn_ref: fn_type
](ImplicitlyCopyable):
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
    fn __del__(deinit self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    fn expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)


@register_passable("trivial")
struct AddressSpace:
    """Address space of the pointer."""

    var _value: Int

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    # CPU address space
    alias GENERIC = AddressSpace(0)

    # GPU address spaces
    alias GLOBAL = AddressSpace(1)
    alias SHARED = AddressSpace(3)
    alias CONSTANT = AddressSpace(4)
    alias LOCAL = AddressSpace(5)
    alias SHARED_CLUSTER = AddressSpace(7)

    @always_inline("builtin")
    fn __mlir_index__(self) -> __mlir_type.index:
        return self._value._mlir_value


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
        address_space._value._mlir_value,
        `>`,
    ]

    var _value: Self._mlir_type

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, _mlir_value: Self._mlir_type):
        self._value = _mlir_value

    @always_inline("nodebug")
    fn __init__(
        out self, *, ref [origin, address_space._value._mlir_value]to: type
    ):
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
        other_type: type_of(Pointer[type, _, address_space]),
    ](self) -> Pointer[
        mut = mut & other_type.origin.mut,
        type=type,
        origin = origin_of(origin, other_type.origin),
        address_space=address_space,
    ]:
        return self._value  # allow lit.ref to convert.


struct Tuple[*element_types: AnyType](ImplicitlyCopyable):
    fn __init__(out self: Tuple[]):
        pass

    @implicit
    fn __init__(out self, *args: *element_types):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, deinit existing: Self):
        pass

    fn __getitem__[i: Int](ref self) -> ref [self] element_types[i]:
        while __mlir_attr.true:
            pass


@register_passable("trivial")
struct UnsafePointer[
    T: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutAnyOrigin],
]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`, T, `,`, address_space._value._mlir_value, `>`
    ]
    var address: Self._mlir_type

    fn __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @implicit
    @always_inline("builtin")
    fn __init__(out self, value: Self._mlir_type):
        self.address = value

    @always_inline("nodebug")
    fn __init__(out self, *, ref [address_space._value._mlir_value]to: T):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

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
        self_origin: ImmutOrigin
    ](ref [self_origin]self, offset: Int = 0) -> ref [
        Origin[True].cast_from[_lit_indirect_origin[self_origin].result],
        address_space,
    ] T:
        while __mlir_attr.true:
            pass


@register_passable("trivial")
struct _StridedRangeIterator(Iterator):
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


trait Iterator(Movable):
    alias Element: AnyType

    fn __has_next__(self) -> Bool:
        ...

    fn __next__(mut self) -> Self.Element:
        ...


fn paramfor_next_iter[
    IteratorType: Iterator & ImplicitlyCopyable
](it: IteratorType) -> IteratorType:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ will return true.  This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    _ = result.__next__()
    return result


fn paramfor_next_value[
    IteratorType: Iterator & ImplicitlyCopyable & ImplicitlyCopyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # __has_next__ will return true.  This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it
    return result.__next__()


struct Optional[T: ImplicitlyCopyable & Movable]:
    fn __del__(deinit self):
        pass

    fn __init__(out self):
        pass

    @implicit
    fn __init__(out self, var value: T):
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

    fn __moveinit__(out self, deinit other: Self):
        pass

    fn value(ref self) -> ref [self] T:
        while True:
            pass


# ===-----------------------------------------------------------------------===#
# rebind
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn rebind[
    src_type: AnyTrivialRegType, //,
    dest_type: AnyTrivialRegType,
](src: src_type) -> dest_type:
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


@always_inline("nodebug")
fn rebind[
    src_type: AnyType, //,
    dest_type: AnyType,
](ref src: src_type) -> ref [src] dest_type:
    lit = __get_mvalue_as_litref(src)
    rebound = rebind[Pointer[dest_type, origin_of(src)]._mlir_type](lit)
    return __get_litref_as_mvalue(rebound)


# ===-----------------------------------------------------------------------===#
# trait downcast
# ===-----------------------------------------------------------------------===#

alias AnyTrait = type_of(AnyType)
alias downcast[_Trait: AnyTrait, T: AnyType] = __mlir_attr[
    `#kgen.downcast<`, T, `> : `, _Trait
]


@always_inline
fn trait_downcast[
    T: AnyTrivialRegType, //, Trait: AnyTrait
](var x: T) -> downcast[Trait, T]:
    return rebind[downcast[Trait, T]](x)


@always_inline
fn trait_downcast[
    T: AnyType, //, Trait: AnyTrait
](ref x: T) -> ref [x] downcast[Trait, T]:
    return rebind[downcast[Trait, T]](x)


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable:
    fn __int__(self) -> Int:
        ...
