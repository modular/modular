# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# FIXME: "string" and "float" are not part of the standard library - they are
# ancient relics of Mojo bringup. These should be removed from stubs.mojo and
# the dependent tests should be migrated off of them.
comptime string = __mlir_type.`!kgen.string`
comptime float = __mlir_type.`!pop.scalar<f64>`


struct _MLIR:
    comptime POPArrayType[
        size: __mlir_type.index, elt_type: AnyType
    ] = __mlir_type[`!pop.array<`, size, `, `, elt_type, `>`]

    comptime KGENParamListType[elt_type: AnyType] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]
    comptime KGENTypeListType[elt_type: type_of(AnyType)] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]


comptime AnyOrigin[*, mut: Bool = False] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.any.origin : !lit.origin<`, +mut._mlir_value, `>`
    ]
]()
comptime ExternalOrigin[*, mut: Bool = False] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.union<> : !lit.origin<`,
        mut._mlir_value,
        `>`,
    ],
]()
comptime StaticConstantOrigin = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.origin.field<`,
        `#lit.static.origin : !lit.origin<0>`,
        `, "__constants__"> : !lit.origin<0>`,
    ]
]()

comptime OriginSet = __mlir_type.`!lit.origin.set`
comptime Never = __mlir_type.`!kgen.never`
comptime EllipsisType = __mlir_type.`!lit.ellipsis`

comptime _lit_origin_type_of_mut[mut: Bool] = __mlir_type[
    `!lit.origin<`, mut._mlir_value, `>`
]


struct Origin[
    mut: Bool = False, _mlir_origin: _lit_origin_type_of_mut[mut], //
](TrivialRegisterPassable):
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    def __init__(v: Origin) -> Origin[mut=False, _mlir_origin=v._mlir_origin]:
        return {}

    @always_inline("builtin")
    @staticmethod
    def unsafe_mut_cast[
        dest_mut: Bool
    ]() -> Origin[
        _mlir_origin=__mlir_attr[
            `#lit.origin.mutcast<`,
            Self._mlir_origin,
            `> : !lit.origin<`,
            dest_mut._mlir_value,
            `>`,
        ],
    ]:
        return {}


comptime _lit_indirect_origin[mut: Bool, //, base: Origin[mut=mut]] = Origin[
    _mlir_origin=__mlir_attr[
        `#lit.indirect.origin<`,
        base._mlir_origin,
        `> : `,
        type_of(base._mlir_origin),
    ]
]()


trait Iterable:
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        ...


# Implement the 'Some' and 'SomeTypeList' helper.
comptime __SomeImpl[Trait: type_of(AnyType), T: Trait] = T
comptime Some[Trait: type_of(AnyType)] = __SomeImpl[Trait, ...]

comptime __SomeTypeListImpl[
    Trait: type_of(AnyType), values: _MLIR.KGENTypeListType[Trait]
] = TypeList[Trait=Trait, values]()
comptime SomeTypeList[Trait: type_of(AnyType)] = __SomeTypeListImpl[Trait, ...]


# ===----------------------------------------------------------------------=== #
# Builtin Types
# ===----------------------------------------------------------------------=== #


comptime KeyElement = Copyable  # & Hashable & Equatable


struct Error(Copyable):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, value: StringLiteral):
        pass

    def __del__(deinit self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # A method for testing.
    def use(self):
        pass


struct NoneType(TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type.`!kgen.none`
    """Raw MLIR type of the `None` value."""

    var _value: Self._mlir_type

    # FIXME: Fix representation of None literal to remove this.
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.`!kgen.none`):
        self._value = value


@stable
@__nonmaterializable(Int)
struct IntLiteral[value: __mlir_type.`!pop.int_literal`](
    TrivialRegisterPassable
):
    comptime _zero = IntLiteral[
        __mlir_attr.`#pop.int_literal<0> : !pop.int_literal`
    ]()
    comptime _one = IntLiteral[
        __mlir_attr.`#pop.int_literal<1> : !pop.int_literal`
    ]()

    @stable
    @always_inline("builtin")
    def __init__(out self):
        """Constructor for any value."""
        pass

    @always_inline("builtin")
    def __ne__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<ne `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __le__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<le `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return self != Self._zero

    @always_inline("builtin")
    def __mul__(
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
    def __sub__(
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
    def __neg__(self) -> type_of(0 - self):
        return 0 - self

    @always_inline("builtin")
    def __pos__(self) -> Self:
        return self

    @always_inline("builtin")
    def __pow__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<pow `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}

    @always_inline("builtin")
    def __floordiv__(
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

    @always_inline("builtin")
    def __xor__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<xor `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}

    @always_inline("builtin")
    def __lshift__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<lshift `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        return {}


@__nonmaterializable(FloatDyn)
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`](
    TrivialRegisterPassable
):
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("builtin")
    @implicit
    def __init__(
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
    def __neg__(self, out result: type_of(self * -1)):
        result = type_of(result)()

    @always_inline("builtin")
    def __mul__(
        self,
        rhs: FloatLiteral,
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<float_literal_bin<mul `,
                Self.value,
                `,`,
                rhs.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        result = type_of(result)()

    @always_inline("builtin")
    def __truediv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<truediv `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        return {}


struct FloatDyn(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.`!pop.scalar<f64>`

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.`!pop.scalar<f64>`):
        self._mlir_value = value

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: FloatLiteral):
        self = __mlir_attr[
            `#pop<float_literal_convert<`, +value.value, `>> : !pop.scalar<f64>`
        ]

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral):
        self = FloatLiteral(value)


@stable
struct Int(Intable, Stringable, TrivialRegisterPassable):
    var _mlir_value: __mlir_type.index

    @stable
    @always_inline("builtin")
    def __init__(out self):
        self._mlir_value = __mlir_op.`index.constant`[
            value=__mlir_attr.`0:index`
        ]()

    @stable
    @always_inline("builtin")
    def __init__(out self, *, mlir_value: __mlir_type.index):
        self._mlir_value = mlir_value

    @stable
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: IntLiteral[_]):
        self._mlir_value = __mlir_attr[
            `#pop.cast_to_builtin<#pop.int_literal_convert<`,
            +value.value,
            `> : !pop.scalar<index>> : index`,
        ]

    @always_inline("builtin")
    def __add__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.add`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __sub__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.sub`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __mul__(lhs, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.mul`(lhs._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Int):
        self = self + rhs

    @always_inline("nodebug")
    def __isub__(mut self, rhs: Int):
        self = self - rhs

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Int):
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Int):
        self = self % rhs

    @always_inline("nodebug")
    def __ipow__(mut self, rhs: Int):
        self = self**rhs

    @always_inline("nodebug")
    def __ilshift__(mut self, rhs: Int):
        self = self << rhs

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Int):
        self = self & rhs

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Int):
        self = self ^ rhs

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Int):
        self = self | rhs

    @always_inline("nodebug")
    def __mod__(self, rhs: Int) -> Int:
        pass

    @always_inline("builtin")
    def __eq__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate eq>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __ne__(self, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate ne>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __lt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate sgt>`
        ](rhs._mlir_value, lhs._mlir_value)

    @always_inline("builtin")
    def __le__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate sle>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __gt__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate sgt>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __ge__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index<cmp_predicate sge>`
        ](lhs._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return not (self == 0)

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._mlir_value

    @always_inline("builtin")
    def __truediv__(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.divs`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __floordiv__(self, rhs: Int) -> Int:
        pass

    @always_inline("builtin")
    def __int__(self) -> Int:
        return self

    @always_inline("nodebug")
    def __pow__(self, exp: Self) -> Self:
        pass

    @always_inline("builtin")
    def __neg__(self) -> Int:
        return self * -1

    @always_inline("builtin")
    def __and__(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.and`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Int):
        self = self * rhs

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Int):
        self = self >> rhs

    @always_inline("nodebug")
    def __rshift__(self, rhs: Int) -> Int:
        pass

    @always_inline("nodebug")
    def __lshift__(self, rhs: Int) -> Int:
        pass

    @always_inline("builtin")
    def __or__(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.or`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __xor__(self, rhs: Int) -> Int:
        return Int(
            mlir_value=__mlir_op.`index.xor`(self._mlir_value, rhs._mlir_value)
        )

    def __str__(self) -> String:
        return "[unimplemented]"


struct UInt8(TrivialRegisterPassable):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, value: IntLiteral):
        pass


comptime Byte = UInt8


struct Span[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
](TrivialRegisterPassable):
    # Field
    var _data: UnsafePointer[Self.T, Self.origin]
    var _len: Int

    def __init__(out self):
        self._data = UnsafePointer[Self.T, Self.origin]()
        self._len = 0

    def __init__(
        out self, *, ptr: UnsafePointer[Self.T, Self.origin], length: Int
    ):
        self._data = ptr
        self._len = length

    @always_inline
    @implicit
    def __init__[U: Copyable](out self, ref[Self.origin] list: List[U]):
        self = {}  # Stub impl.

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Self.T, Self.origin]:
        return self._data

    @always_inline
    def __getitem__(self, idx: Int) -> ref[Self.origin] Self.T:
        return self._data[normalized_idx]


@stable
@__nonmaterializable(String)
struct StringLiteral[value: __mlir_type.`!kgen.string`](
    TrivialRegisterPassable
):
    @stable
    @always_inline("builtin")
    def __init__(out self):
        pass

    @always_inline("nodebug")
    def __eq__(self, other: StringLiteral) -> Bool:
        return Bool()

    # TODO(MSTDL-1327): Reduce pain when string literals can't be
    # nonmaterializable by making them merge into StaticString.  They should
    # eventually merge into String through nonmaterialization.
    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(StringLiteral[_]),
    ](self) -> StaticString:
        return self

    @always_inline("nodebug")
    def __add__(
        self, rhs: StringLiteral
    ) -> StringLiteral[
        __mlir_attr[
            `#pop.string_concat<`,
            self.value,
            `,`,
            rhs.value,
            `> : !kgen.string`,
        ]
    ]:
        return {}


struct StringSlice[mut: Bool, //, origin: Origin[mut=mut]](
    TrivialRegisterPassable
):
    var _slice: Span[Byte, Self.origin]

    @implicit
    def __init__[](out self, ref[Self.origin] value: String):
        self._slice = {}

    @implicit
    def __init__(out self: StaticString, lit: StringLiteral):
        pass

    @always_inline
    def __init__(out self: StaticString, _kgen: __mlir_type.`!kgen.string`):
        pass

    @always_inline
    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Byte, origin]:
        return self._slice.unsafe_ptr()

    @always_inline
    def byte_length(self) -> Int:
        return self._slice._len


comptime StaticString = StringSlice[StaticConstantOrigin]


@always_inline("builtin")
def _get_kgen_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.values,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
def get_static_string[
    string: StaticString, *extra: StaticString
]() -> StaticString:
    return StaticString(_get_kgen_string[string, *extra]())


trait Stringable:
    def __str__(self) -> String:
        ...


struct String(ImplicitlyCopyable, KeyElement):
    def __init__(out self):
        pass

    @implicit
    def __init__(out self, literal: StringLiteral):
        pass

    def __init__[T: Stringable](out self, value: T):
        self = value.__str__()

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: String):
        pass

    def __del__(deinit self):
        pass

    def __len__(self) -> Int:
        return 0

    def __contains__(self, substr: StringSlice[mut=False, ...]) -> Bool:
        return True

    def __add__(self, other: StringSlice) -> String:
        pass

    def __iadd__(mut self, rhs: StringSlice[mut=False, ...]):
        pass

    def byte_length(self) -> Int:
        return 0

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[UInt8, origin_of(self)]:
        return {}


@stable
struct Bool(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.i1

    @stable
    @always_inline("builtin")
    def __init__(out self):
        self._mlir_value = __mlir_attr.`0 : i1`

    @stable
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.i1):
        self._mlir_value = value

    @always_inline("builtin")
    def __mlir_i1__(self) -> __mlir_type.i1:
        return self._mlir_value

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        return self

    @always_inline("builtin")
    def __invert__(self) -> Bool:
        return __mlir_op.`pop.xor`(self._mlir_value, __mlir_attr.true)

    @always_inline("builtin")
    def __and__(self, rhs: Bool) -> Bool:
        return __mlir_op.`pop.and`(self._mlir_value, rhs._mlir_value)


struct Slice(TrivialRegisterPassable):
    @implicit
    def __init__(out self, end: Int):
        pass

    def __init__(out self, start: Int, end: Int):
        return

    def __init__[
        T0: TrivialRegisterPassable,
        T1: TrivialRegisterPassable,
        T2: TrivialRegisterPassable,
    ](
        out self,
        start: T0,
        end: T1,
        step: T2,
        __slice_literal__: NoneType = None,
    ):
        pass


@fieldwise_init
struct _ListIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    comptime Element = Self.T  # FIXME(MOCO-2068): shouldn't be needed.

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[List[Self.Element], Self.origin]

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        abort()


struct List[T: Copyable](Copyable, Iterable):
    def __init__(out self):
        pass

    def __init__(out self, var *elements: Self.T, __list_literal__: NoneType):
        pass

    def append(mut self, var value: Self.T):
        pass

    def __getitem__(ref self, idx: Int) -> ref[self] Self.T:
        pass

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ListIter[Self.T, iterable_origin, True]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        abort()


struct Set[T: AnyType]:
    def __init__(out self, *elements: Self.T, __set_literal__: NoneType = None):
        pass

    def add(mut self, var value: Self.T):
        pass


struct Dict[
    K: Copyable & ImplicitlyDestructible, V: Copyable & ImplicitlyDestructible
]:
    def __init__(out self):
        pass

    def __init__(
        out self,
        var keys: List[Self.K],
        var values: List[Self.V],
        __dict_literal__: NoneType,
    ):
        pass

    def __setitem__(mut self, key: Self.K, value: Self.V):
        pass


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


# A linear type, see
# https://www.notion.so/modularai/Linear-Types-14a1044d37bb809ab074c990fe1a84e3.
trait AnyType:
    pass


@explicit_destroy
trait Copyable(Movable):
    def __init__(out self, *, copy: Self):
        ...

    @always_inline
    def copy(self) -> Self:
        return Self(copy=self)

    comptime __copy_ctor_is_trivial: Bool


trait ImplicitlyCopyable(Copyable, ImplicitlyDestructible):
    pass


trait TrivialRegisterPassable(
    ImplicitlyCopyable, ImplicitlyDestructible, Movable, RegisterPassable
):
    pass


trait RegisterPassable(Movable):
    pass


trait Defaultable(ImplicitlyDestructible):
    def __init__(out self):
        ...


def materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile time parameter into a runtime value."""
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


@explicit_destroy
trait ExplicitlyDestroyedMovable:
    def __init__(out self, *, deinit take: Self):
        ...


@explicit_destroy
trait Movable:
    def __init__(out self, *, deinit take: Self):
        ...

    comptime __move_ctor_is_trivial: Bool


trait ImplicitlyDestructible:
    def __del__(deinit self, /):
        ...

    comptime __del__is_trivial: Bool


# ===-----------------------------------------------------------------------===#
# ParameterList
# ===-----------------------------------------------------------------------===#


struct Variadic:
    comptime TypesOfTrait[T: type_of(AnyType)] = _MLIR.KGENTypeListType[T]

    # ===-----------------------------------------------------------------------===#
    # Utils
    # ===-----------------------------------------------------------------------===#

    comptime types[T: type_of(AnyType), //, *Ts: T] = Ts.values


struct ParameterList[type: AnyType, //, values: _MLIR.KGENParamListType[type]](
    TrivialRegisterPassable
):
    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `> : index`,
        ]
    )

    def __init__(out self):
        pass

    @always_inline
    def __getitem__(self, idx: Int) -> Self.type:
        pass

    comptime __getitem_param__[idx: Int]: Self.type = __mlir_attr[
        `#kgen.param_list.get<:`,
        type_of(Self.values),
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.type,
    ]

    comptime tabulate[
        type: AnyType,
        //,
        count: Int,
        Mapper: _TabulateIntToValueGeneratorType[type],
    ] = ParameterList[
        type=type,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count._mlir_value,
            `,`,
            _IndexToIntTabulateWrap[Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[type],
        ],
    ]

    comptime splat[T: AnyType, //, count: Int, value: T] = ParameterList[
        T,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count._mlir_value,
            `,`,
            _IndexToIntTabulateWrap[_SplatValueTabulator[value, _], ...],
            `> : `,
            _MLIR.KGENParamListType[T],
        ],
    ]


comptime _TabulateIntToValueGeneratorType[ToT: AnyType] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `>`,
    +ToT,
    `>`,
]

comptime _IndexToIntTabulateWrap[
    ToT: AnyType,
    //,
    ToWrap: _TabulateIntToValueGeneratorType[ToT],
    idx: __mlir_type.index,
]: ToT = ToWrap[Int(mlir_value=idx)]


# ===-----------------------------------------------------------------------===#
# TypeList
# ===-----------------------------------------------------------------------===#


struct TypeList[
    Trait: type_of(AnyType), //, values: _MLIR.KGENTypeListType[Trait]
](TrivialRegisterPassable):
    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `> : index`,
        ]
    )

    comptime __getitem_param__[idx: Int]: Self.Trait = __mlir_attr[
        `#kgen.param_list.get<:`,
        type_of(Self.values),
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.Trait,
    ]

    @always_inline("builtin")
    def __init__(out self):
        pass

    @implicit
    @always_inline("builtin")
    def __init__(
        existing: TypeList[...],
        out self: TypeList[
            Trait=Self.Trait,
            __mlir_attr[
                `#kgen.upcast<`,
                existing.values,
                `> : `,
                _MLIR.KGENTypeListType[Self.Trait],
            ],
        ],
    ) where __mlir_attr[
        `#kgen.is_refined_type<`, existing.Trait, `, `, Self.Trait, `>`
    ]:
        pass

    comptime tabulate[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        count: Int,
        Mapper: _TabulateIntToTypeGeneratorType[Trait, ToT],
    ] = TypeList[
        Trait=Trait,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count._mlir_value,
            `,`,
            _IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
            `> : `,
            _MLIR.KGENTypeListType[Trait],
        ],
    ]

    comptime splat[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ] = TypeList.tabulate[
        Trait=Trait, ToT=type, count, _SplatTypeTabulator[Trait, type, _]
    ]


comptime _SplatTypeTabulator[
    Trait: type_of(AnyType), T: Trait, index: Int
]: Trait = T


comptime _TabulateIntToTypeGeneratorType[
    Trait: type_of(AnyType), ToT: Trait
] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `> `,
    Trait,
    `>`,
]

comptime _IndexToIntTypeTabulateWrap[
    Trait: type_of(AnyType),
    ToT: Trait,
    //,
    ToWrap: _TabulateIntToTypeGeneratorType[Trait, ToT],
    idx: __mlir_type.index,
] = ToWrap[Int(mlir_value=idx)]


@fieldwise_init
struct _VariadicListIter[
    elt_is_mutable: Bool,
    //,
    elt_type: AnyType,
    elt_origin: Origin[mut=elt_is_mutable],
    list_origin: Origin[mut=False],
    is_owned: Bool,
](RegisterPassable):
    """Iterator for VariadicList.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicList.
    """

    comptime variadic_list_type = VariadicList[
        origin=Self.elt_origin,
        Self.elt_type,
        Self.is_owned,
    ]

    var index: Int
    var src: Pointer[Self.variadic_list_type, Self.list_origin]

    def __init__(
        out self,
        index: Int,
        ref[Self.list_origin] list: Self.variadic_list_type,
    ):
        self.index = index
        self.src = Pointer(to=list)

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.elt_origin] Self.elt_type:
        raise StopIteration()


struct VariadicList[
    elt_is_mutable: Bool,
    # NOTE: origin._mlir_origin is here in the param list.
    origin: Origin[mut=elt_is_mutable],
    //,
    element_type: AnyType,
    is_owned: Bool,
](RegisterPassable):
    comptime _EltPointerType = Pointer[Self.element_type, Self.origin]
    # FIXME: This should be the origin of the container, not ExternalOrigin.
    var value: Span[Self._EltPointerType, ExternalOrigin[]]

    # TODO: the origin of the vardecl is captured in the Self.origin set
    # by the compiler to make sure the container outlives all its elements.
    @implicit
    def __init__[
        size: __mlir_type.index, container_origin: Origin[mut=False]
    ](
        out self,
        value: Pointer[
            _MLIR.POPArrayType[size, Self._EltPointerType._mlir_type],
            container_origin,
        ]._mlir_type,
    ):
        # Convert the !lit.ref to an UnsafePointer, then cast to a pointer to
        # the first element.
        var array_up = UnsafePointer(to=Pointer(value)[])
        var elt_ptr = UnsafePointer[_, ExternalOrigin[]](
            __mlir_op.`pop.array.gep`(
                array_up.address,
                Int(0)._mlir_value,
            )
        ).bitcast[Self._EltPointerType]()
        var size_tmp = size  # FIXME: Weird MLIR syntax error?
        self.value = Span(ptr=elt_ptr, length=Int(mlir_value=size_tmp))

    def __getitem__[
        self_origin: Origin[mut=False]
    ](ref[self_origin] self, idx: Int) -> ref[
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        origin_of(Self.origin, self_origin).unsafe_mut_cast[
            Self.elt_is_mutable
        ]()
    ] Self.element_type:
        while True:
            pass

    def __iter__[
        self_origin: Origin[]
    ](ref[self_origin] self) -> _VariadicListIter[
        Self.element_type, Self.origin, self_origin, Self.is_owned
    ]:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return {0, self}


struct VariadicPack[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    element_trait: type_of(AnyType),
    //,
    is_owned: Bool,
    *element_types: element_trait,
](RegisterPassable):
    comptime _mlir_pack_type = __mlir_type[
        `!lit.ref.pack<:param_list<`,
        Self.element_trait,
        `> `,
        Self.element_types.values,
        `, `,
        Self.origin._mlir_origin,
        `>`,
    ]
    var _value: Self._mlir_pack_type

    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_disable_nested_origin_exclusivity
    def __init__(out self, value: Self._mlir_pack_type):
        self._value = value

    def __getitem_param__[
        index: Int
    ](self) -> ref[Self.origin] Self.element_types[index]:
        while True:
            pass


struct AddressSpace(TrivialRegisterPassable):
    """Address space of the pointer."""

    var _value: Int

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Int):
        self._value = value

    # CPU address space
    comptime GENERIC = AddressSpace(0)

    # GPU address spaces
    comptime GLOBAL = AddressSpace(1)
    comptime SHARED = AddressSpace(3)
    comptime CONSTANT = AddressSpace(4)
    comptime LOCAL = AddressSpace(5)
    comptime SHARED_CLUSTER = AddressSpace(7)

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        return self._value._mlir_value


struct Pointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    address_space: AddressSpace = AddressSpace.GENERIC,
](TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[
        `!lit.ref<`,
        Self.type,
        `, `,
        Self.origin._mlir_origin,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var _value: Self._mlir_type

    @always_inline("nodebug")
    @implicit
    def __init__(out self, _mlir_value: Self._mlir_type):
        self._value = _mlir_value

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(_mlir_value=__get_mvalue_as_litref(to))

    @staticmethod
    @always_inline("nodebug")
    def address_of(
        ref[Self.origin, Self.address_space] value: Self.type
    ) -> Self:
        return Pointer(_mlir_value=__get_mvalue_as_litref(value))

    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        return __get_litref_as_mvalue(self._value)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    def __eq__(
        self, rhs: Pointer[Self.type, _, Self.address_space, ...]
    ) -> Bool:
        return True

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(Pointer[Self.type, _, Self.address_space]),
    ](self) -> Pointer[
        mut=Self.mut & other_type.origin.mut,
        type=Self.type,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
    ]:
        return {self._value}  # allow lit.ref to convert.


struct Tuple[*element_types: Movable](ImplicitlyCopyable):
    comptime _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        _MLIR.KGENTypeListType[Movable],
        Self.element_types.values,
        `>`,
    ]
    var _mlir_value: Self._mlir_type

    def __init__(out self: Tuple[]):
        pass

    @implicit
    def __init__(out self, *args: *Self.element_types):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __getitem_param__[i: Int](ref self) -> ref[self] Self.element_types[i]:
        while __mlir_attr.true:
            pass


struct UnsafePointer[
    mut: Bool,
    //,
    type: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](Defaultable, TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `,`,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    var address: Self._mlir_type

    def __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @implicit
    @always_inline("builtin")
    def __init__(out self, value: Self._mlir_type):
        self.address = value

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.type,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @staticmethod
    def address_of(ref[Self.address_space] arg: Self.type) -> Self:
        return Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(arg)))

    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.type:
        while __mlir_attr.true:
            pass

    def __getitem__(
        self, offset: Int
    ) -> ref[Self.origin, Self.address_space] Self.type:
        while __mlir_attr.true:
            pass

    # This returns a reference to an element with an origin specified by as a
    # unique reference from this pointer.  The returned reference is always
    # mutable.
    def get_unique_item_ref[
        self_origin: Origin[]
    ](ref[self_origin] self, offset: Int = 0) -> ref[
        _lit_indirect_origin[self_origin].unsafe_mut_cast[True](),
        Self.address_space,
    ] Self.type:
        while __mlir_attr.true:
            pass

    @always_inline
    def take_pointee[
        T: Movable, //
    ](self: UnsafePointer[T, _]) -> T where type_of(self).mut:
        return __get_address_as_owned_value(self.address)

    @always_inline("builtin")
    def bitcast[
        T: AnyType
    ](self) -> UnsafePointer[T, Self.origin, address_space=Self.address_space]:
        return __mlir_op.`pop.pointer.bitcast`[
            _type=UnsafePointer[
                T,
                Self.origin,
                address_space=Self.address_space,
            ]._mlir_type,
        ](self.address)

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = UnsafePointer[
        Self.type,
        target_origin,
        address_space=Self.address_space,
    ]

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        return __mlir_op.`pop.pointer.bitcast`[
            _type=Self._OriginCastType[target_origin]._mlir_type,
        ](self.address)


comptime MutOpaquePointer[
    origin: Origin[mut=True],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointer[NoneType, origin, address_space=address_space]


struct _StridedRangeIterator(Iterator, TrivialRegisterPassable):
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    def __len__(self) -> Int:
        if self.step > 0 and self.start < self.end:
            return self.end - self.start
        elif self.step < 0 and self.start > self.end:
            return self.start - self.end
        else:
            return 0

    @always_inline
    def __next__(mut self) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result = self.start
        self.start += self.step
        return result


# ===-----------------------------------------------------------------------===#
# parameter_for
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct StopIteration(TrivialRegisterPassable):
    pass


trait Iterator(ImplicitlyDestructible, Movable):
    comptime Element: Movable

    def __next__(mut self) raises StopIteration -> Self.Element:
        ...


def paramfor_has_next[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> Bool where conforms_to(
    IteratorType.Element,
    Movable & ImplicitlyDestructible,
):
    var result = it.copy()
    try:
        var elem = result.__next__()
        _ = trait_downcast_var[Movable & ImplicitlyDestructible](elem^)
        return True
    except:
        return False


def paramfor_next_iter[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType where conforms_to(
    IteratorType.Element,
    Movable & ImplicitlyDestructible,
):
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    # This intentionally discards the value, but this only happens at comptime,
    # so recomputing it in the body of the loop is fine.
    try:
        var elem = result.__next__()
        _ = trait_downcast_var[Movable & ImplicitlyDestructible](elem^)
    except:
        abort()
    return result^


def paramfor_next_value[
    IteratorType: Iterator & Copyable
](it: IteratorType) -> IteratorType.Element:
    # NOTE: This function is called by the compiler's elaborator only when
    # paramfor_has_next will return true. This is needed because the interpreter
    # memory model isn't smart enough to handle mut arguments cleanly.
    var result = it.copy()
    try:
        return result.__next__()
    except:
        abort()


struct Optional[T: Movable](Copyable):
    def __del__(deinit self):
        pass

    def __init__(out self):
        pass

    @implicit
    def __init__(out self, var value: Self.T):
        # This isn't correct impl, but silences an error.
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(value))

    @implicit
    def __init__(out self, value: NoneType):
        pass

    # FIXME: None literal should be of NoneType not !kgen.none.
    @implicit
    def __init__(out self, x: __mlir_type.`!kgen.none`):
        pass

    def value(ref self) -> ref[self] Self.T:
        while True:
            pass


# ===-----------------------------------------------------------------------===#
# rebind
# ===-----------------------------------------------------------------------===#


@always_inline("builtin")
def rebind[
    src_type: TrivialRegisterPassable,
    //,
    dest_type: TrivialRegisterPassable,
](src: src_type) -> dest_type:
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


@always_inline("nodebug")
def rebind[
    src_type: AnyType,
    //,
    dest_type: AnyType,
](ref src: src_type) -> ref[src] dest_type:
    lit = __get_mvalue_as_litref(src)
    rebound = rebind[Pointer[dest_type, origin_of(src)]._mlir_type](lit)
    return __get_litref_as_mvalue(rebound)


def rebind_var[
    src_type: Movable,
    //,
    dest_type: Movable,
](var src: src_type, out dest: dest_type):
    ref dest_ref = rebind[dest_type](src)
    dest = UnsafePointer(to=dest_ref).take_pointee()
    __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(src))


# ===-----------------------------------------------------------------------===#
# trait downcast
# ===-----------------------------------------------------------------------===#

comptime downcast[T: AnyType, _Trait: type_of(AnyType)] = __mlir_attr[
    `#kgen.downcast<:`, type_of(T), +T, `> : `, _Trait
]


@always_inline
def trait_downcast[
    T: TrivialRegisterPassable, //, Trait: type_of(AnyType)
](var src: T) -> downcast[T, Trait]:
    return rebind[downcast[T, Trait]](src)


def trait_downcast_var[
    T: Movable,
    //,
    Trait: type_of(Movable),
](var src: T) -> downcast[T, Trait]:
    return rebind_var[downcast[T, Trait]](src^)


@always_inline
def trait_downcast[
    T: AnyType, //, Trait: type_of(AnyType)
](ref x: T) -> ref[x] downcast[T, Trait]:
    return rebind[downcast[T, Trait]](x)


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable(ImplicitlyDestructible):
    def __int__(self) -> Int:
        ...


# ===----------------------------------------------------------------------=== #
#  DType
# ===----------------------------------------------------------------------=== #


struct DType(TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type.`!kgen.dtype`
    var _mlir_value: Self._mlir_type

    comptime int = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`
    )
    comptime float32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    )
    comptime float64 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`
    )
    comptime int32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    )
    comptime uint32 = DType(
        mlir_value=__mlir_attr.`#kgen.dtype.constant<ui32> : !kgen.dtype`
    )

    @always_inline("builtin")
    @implicit
    def __init__(out self, mlir_value: Self._mlir_type):
        self._mlir_value = mlir_value


comptime Float32 = SIMD[DType.float32, 1]
comptime Float64 = SIMD[DType.float64, 1]
comptime Int32 = SIMD[DType.int32, 1]
comptime UInt32 = SIMD[DType.uint32, 1]

# ===----------------------------------------------------------------------=== #
#  SIMD
# ===----------------------------------------------------------------------=== #


struct SIMD[dtype: DType, size: Int](TrivialRegisterPassable):
    comptime _mlir_type = __mlir_type[
        `!pop.simd<`, Self.size._mlir_value, `, `, Self.dtype._mlir_value, `>`
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the vector."""

    @always_inline("nodebug")
    def __init__(out self, *, mlir_value: Self._mlir_type):
        self._mlir_value = mlir_value

    @always_inline("nodebug")
    def __init__(out self):
        comptime res = SIMD[Self.dtype, Self.size](Int())
        self = res

    @implicit
    @always_inline
    def __init__(out self, value: Int, /):
        var index = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!pop.scalar<index>`
        ](value._mlir_value)
        var s = __mlir_op.`pop.cast`[_type=SIMD[Self.dtype, 1]._mlir_type](
            index
        )

        comptime if Self.size == 1:
            self._mlir_value = rebind[Self._mlir_type](s)
        else:
            self._mlir_value = __mlir_op.`pop.simd.splat`[
                _type=Self._mlir_type
            ](s)

    @implicit
    def __init__(out self, value: FloatLiteral, /):
        var res = __mlir_attr[
            `#pop<float_literal_convert<`, value.value, `>> : `, Self._mlir_type
        ]
        self = Self(mlir_value=res)

    def __add__(lhs, rhs: Self) -> Self:
        while __mlir_attr.true:
            pass

    @staticmethod
    def splat():
        pass

    @always_inline("nodebug")
    def __truediv__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.div`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __rtruediv__(self, value: Self) -> Self:
        return value / self

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline("nodebug")
    def join(self, other: Self) -> SIMD[Self.dtype, 2 * Self.size]:
        return SIMD[Self.dtype, 2 * Self.size]()


@no_inline
def abort() -> Never:
    __mlir_op.`llvm.intr.trap`()
    while True:
        pass
