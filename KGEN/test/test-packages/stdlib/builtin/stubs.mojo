# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias int = __mlir_type.index
alias string = __mlir_type.`!kgen.string`
alias float = __mlir_type.`!pop.scalar<f64>`

alias NoneType = __mlir_type.`!kgen.none`
alias AnyTrivialRegType = __mlir_type.`!kgen.type`

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


struct AnyLifetime[is_mutable: __mlir_type.i1]:
    """This represents a lifetime reference of potentially parametric type.
    TODO: This should be replaced with a parametric type alias.

    Parameters:
        is_mutable: Whether the lifetime reference is mutable.
    """

    alias type = __mlir_type[
        `!lit.lifetime<`,
        is_mutable,
        `>`,
    ]


# ===----------------------------------------------------------------------=== #
# Builtin Types
# ===----------------------------------------------------------------------=== #


trait CollectionElement:
    pass


trait KeyElement:
    pass


@register_passable
struct Error:
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass


struct object:
    fn __init__(inout self):
        pass

    fn __init__(inout self, value: NoneType):
        pass

    fn __init__(inout self, value: Int):
        pass

    fn __copyinit__(inout self, existing: Self, /):
        pass


@value
@nonmaterializable(Int)
@register_passable("trivial")
struct IntLiteral:
    var value: __mlir_type.`!kgen.int_literal`


@value
@nonmaterializable(FloatDyn)
@register_passable("trivial")
struct FloatLiteral:
    var value: __mlir_type.`!kgen.float_literal`


@value
@register_passable("trivial")
struct FloatDyn:
    var value: __mlir_type.f64

    @always_inline("nodebug")
    fn __init__(inout self, value: FloatLiteral):
        self = Self(
            __mlir_op.`kgen.float_literal.convert`[_type = __mlir_type.f64](
                value.value
            )
        )


@value
@register_passable("trivial")
struct Int(Copyable):
    var value: int

    @always_inline("nodebug")
    fn __init__(inout self, value: IntLiteral):
        self.value = __mlir_op.`kgen.int_literal.convert`[
            _type = __mlir_type.index
        ](value.value)

    @always_inline("nodebug")
    fn __add__(lhs, rhs: Int) -> Int:
        return __mlir_op.`index.add`(lhs.value, rhs.value)

    @always_inline("nodebug")
    fn __iadd__(inout self, rhs: Int):
        self = self + rhs

    @always_inline("nodebug")
    fn __eq__(lhs, rhs: Int) -> Bool:
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](lhs.value, rhs.value)

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return not (self == 0)

    @always_inline("nodebug")
    fn __index__(self) -> Int:
        return self

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        return self.value


@value
@register_passable("trivial")
struct StringLiteral:
    var value: __mlir_type.`!kgen.string`

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return Bool()


struct String(KeyElement):
    fn __init__(inout self, literal: StringLiteral):
        pass


@value
@register_passable("trivial")
struct Bool(AnyType):
    var value: __mlir_type.i1

    @always_inline("nodebug")
    fn __init__(inout self):
        self.value = __mlir_attr.`0 : i1`

    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.i1):
        self.value = value

    fn __mlir_i1__(self) -> __mlir_type.i1:
        return self.value

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        return self

    @always_inline("nodebug")
    fn __invert__(self) -> Bool:
        return self  # Incorrect impl


@register_passable("trivial")
struct Slice:
    fn __init__(inout self, end: int):
        pass

    fn __init__(inout self, start: int, end: int):
        return

    fn __init__[
        T0: AnyTrivialRegType, T1: AnyTrivialRegType, T2: AnyTrivialRegType
    ](start: T0, end: T1, step: T2) -> Self:
        return Self {}


# ===----------------------------------------------------------------------=== #
# Value Stubs
# ===----------------------------------------------------------------------=== #


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        pass


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        pass


trait AnyType:
    fn __del__(owned self, /):
        ...


# ===----------------------------------------------------------------------=== #
# Coroutine Stubs
# ===----------------------------------------------------------------------=== #


@value
@register_passable
struct Coroutine[T: AnyType, lifetimes: __mlir_type.`!lit.lifetime.set`]:
    var value: __mlir_type.`!co.routine`

    fn __await__(self) -> T:
        while __mlir_attr.true:
            pass


@value
@register_passable
struct RaisingCoroutine[T: AnyType, lifetimes: __mlir_type.`!lit.lifetime.set`]:
    var value: __mlir_type.`!co.routine`

    fn __await__(self) raises -> T:
        while __mlir_attr.true:
            pass


# ===----------------------------------------------------------------------=== #
# Builtin Collection Stubs
# ===----------------------------------------------------------------------=== #


@register_passable
struct VariadicList[type: AnyTrivialRegType]:
    alias _mlir_type = __mlir_type[`!kgen.variadic<`, type, `>`]

    fn __init__(inout self, value: Self._mlir_type):
        return


struct VariadicListMem[
    element_type: AnyType,
    elt_is_mutable: __mlir_type.i1,
    lifetime: __mlir_type[`!lit.lifetime<`, elt_is_mutable, `>`],
]:
    alias _mlir_type = __mlir_type[
        `!lit.ref<`, element_type, `, `, lifetime, `, 0>`
    ]

    fn __init__(
        inout self,
        value: __mlir_type[
            `!kgen.variadic<`, Self._mlir_type, `, borrow_in_mem>`
        ],
    ):
        pass

    fn __init__(
        inout self,
        value: __mlir_type[`!kgen.variadic<`, Self._mlir_type, `, inout>`],
    ):
        pass

    fn __init__(
        inout self,
        value: __mlir_type[
            `!kgen.variadic<`, Self._mlir_type, `, owned_in_mem>`
        ],
    ):
        pass


alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


@register_passable
struct VariadicPack[
    elt_is_mutable: __mlir_type.i1,
    lifetime: __mlir_type[`!lit.lifetime<`, elt_is_mutable, `>`],
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
]:
    alias _mlir_pack_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        lifetime,
        `>`,
    ]

    fn __init__(inout self, value: Self._mlir_pack_type, is_owned: Bool):
        pass


@register_passable
struct __ParameterClosureCaptureList[
    fn_type: AnyTrivialRegType, fn_ref: fn_type
]:
    var value: __mlir_type.`!kgen.pointer<none>`

    # Parameter closure invariant requires this function be marked 'capturing'.
    @parameter
    @always_inline
    fn __init__(inout self):
        self.value = __mlir_op.`kgen.capture_list.create`[callee=fn_ref]()

    @always_inline
    fn __copyinit__(inout self, existing: Self):
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

    alias GENERIC = AddressSpace(0)

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        return self._value.value


@value
@register_passable("trivial")
struct Reference[
    is_mutable: __mlir_type.i1, //,
    type: AnyType,
    lifetime: AnyLifetime[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        lifetime,
        `, `,
        address_space._value.value,
        `>`,
    ]

    fn __init__(inout self, value: Self._mlir_type):
        pass

    fn __getitem__(self) -> ref [lifetime, address_space] type:
        while __mlir_attr.true:
            pass


struct Tuple[*element_types: AnyType]:
    fn __init__(inout self, *args: *element_types):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    fn __getitem__[
        i: Int
    ](ref [_]self: Self) -> ref [__lifetime_of(self)] element_types[i.value]:
        while __mlir_attr.true:
            pass


@register_passable("trivial")
struct UnsafePointer[
    T: AnyType, address_space: AddressSpace = AddressSpace.GENERIC
]:
    alias _ref_lifetime = __mlir_attr.`#lit.lifetime<1>: !lit.lifetime<1>`

    fn __getitem__(
        self,
    ) -> ref [Self._ref_lifetime, address_space._value.value] T:
        while __mlir_attr.true:
            pass

    fn __getitem__(
        self, offset: Int
    ) -> ref [Self._ref_lifetime, address_space._value.value] T:
        while __mlir_attr.true:
            pass
