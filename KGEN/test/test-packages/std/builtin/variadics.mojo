# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

comptime VariadicOf[T: type_of(AnyType)] = __mlir_type[
    `!kgen.variadic<`, T, `>`
]

# Some magic to create comptime variadics without introducing a kgen.variadic.create attr.
comptime EmptyVariadic[T: type_of(AnyType)] = __mlir_attr[
    `#kgen.variadic<>: !kgen.variadic<`, T, `>`
]
comptime MakeVariadic[T: type_of(AnyType), //, *Ts: T] = Ts

comptime VariadicConcat[
    T: type_of(AnyType), //, *Ts: VariadicOf[T]
] = __mlir_attr[`#kgen.variadic.concat<`, Ts, `> :`, VariadicOf[T]]

# This specifies a generator to generate a generator type for the reducer of
# [Prev: AnyType, Ts: VariadicOf[AnyType], idx :Int] -> Prev
comptime ReduceVariadicIdxGeneratorTypeGenerator[
    Prev: AnyType, From: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    __mlir_type.index,
    `>`,
    +Prev,
    `>`,
]

# This specifies a generator to generate a generator type for the mapper of
# [Ts: VariadicOf[AnyType], idx :Int] -> AnyType
comptime MapVariadicIdxToTypeGeneratorTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    __mlir_type.index,
    `>`,
    To,
    `>`,
]

# This create a reducer out of a mapper.
comptime WrapVariadicIdxToTypeMapperToReducer[
    F: type_of(AnyType),
    T: type_of(AnyType),
    Mapper: MapVariadicIdxToTypeGeneratorTypeGenerator[F, T],
    Prev: VariadicOf[T],
    From: VariadicOf[F],
    Idx: __mlir_type.index,
] = VariadicConcat[Prev, MakeVariadic[Mapper[From, Idx]]]


# This specifies a generator to generate a generator type for the mapper of
# [T: AnyType] -> AnyType
comptime MapTypeToTypeGeneratorTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[`!lit.generator<<"From" :`, From, `>`, To, `>`]

comptime TypeToTypeWrap[
    From: type_of(AnyType),
    To: type_of(AnyType),
    ToWrap: MapTypeToTypeGeneratorTypeGenerator[From, To],
    VA: VariadicOf[From],
    idx: __mlir_type.index,
] = ToWrap[VA[idx]]

# Reduce a variadic to a variadic
comptime ReduceVariadicAndIdxToVariadic[
    From: type_of(AnyType),
    To: type_of(AnyType),
    //,
    *,
    Variadic: VariadicOf[From],
    Reducer: ReduceVariadicIdxGeneratorTypeGenerator[VariadicOf[To], From],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    EmptyVariadic[To],  # base
    `,`,
    Variadic,
    `,`,
    Reducer,
    `> : `,
    type_of(EmptyVariadic[To]),
]

comptime MapVariadicAndIdxToType[
    From: type_of(AnyType),
    //,
    *,
    To: type_of(AnyType),
    Variadic: VariadicOf[From],
    Mapper: MapVariadicIdxToTypeGeneratorTypeGenerator[From, To],
] = ReduceVariadicAndIdxToVariadic[
    Variadic=Variadic,
    Reducer = WrapVariadicIdxToTypeMapperToReducer[From, To, Mapper],
]

comptime MapTypeToType[
    From: type_of(AnyType),
    //,
    *,
    To: type_of(AnyType),
    Variadic: __mlir_type[`!kgen.variadic<`, From, `>`],
    Mapper: MapTypeToTypeGeneratorTypeGenerator[From, To],
] = MapVariadicAndIdxToType[
    To=To, Variadic=Variadic, Mapper = TypeToTypeWrap[From, To, Mapper]
]

comptime VariadicZip[T: type_of(AnyType), //, *Ts: VariadicOf[T]] = __mlir_attr[
    `#kgen.variadic.zip<`, Ts, `> : !kgen.variadic<`, VariadicOf[T], `>`
]

# This maps a !variadic<!variadic<*elt>> to !variadic<Tuple[*elt]>
# We can not reuse the `MapTypeToType` defined above because this is, technically,
# a `value-to-type map`, since we are mapping a variadic of type values to a tuple.
# We might want to consider to generalized it to a `MapValueToType`, but for now,
# mapping variadic to tuple is the only use case.
comptime VariadicToTuple[
    From: type_of(AnyType),
    To: type_of(AnyType),
    //,
    VariadicToTypeGen: __mlir_type[
        `!lit.generator<<"Base": `,
        VariadicOf[To],
        `, "From": !kgen.variadic<!kgen.variadic<`,
        From,
        `>>, "Idx":`,
        __mlir_type.index,
        `>`,
        VariadicOf[To],
        `>`,
    ],
    *Ts: VariadicOf[From],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    EmptyVariadic[To],  # base
    `,`,
    Ts,
    `,`,
    VariadicToTypeGen,
    `> : `,
    __mlir_type[`!kgen.variadic<`, To, `>`],
]

comptime VariadicIdxToTupleReducer[
    T: type_of(AnyType),
    Prev: VariadicOf[
        AnyType  # AnyType is the trait_bound_of(Tuple) in test package
    ],
    *Ts: VariadicOf[T],
    idx: __mlir_type.index,
] = VariadicConcat[Prev, MakeVariadic[Tuple[*Ts[idx]]]]


# This is all user need to learn.
comptime ZipToTuple[
    T: type_of(AnyType), //, *Ts: VariadicOf[T]
] = VariadicToTuple[
    # Give a common bound for all the tuple produced. In test package, we can use AnyType.
    # In stdlib, this should be all the traits that Tuple conforms_to.
    VariadicIdxToTupleReducer[T],
    *VariadicZip[*Ts],  # zipped variadic
]

comptime VariadicSplat[T: AnyType, count: Int] = __mlir_attr[
    `#kgen.variadic.splat<`,
    T,
    `,`,
    count._mlir_value,
    `> : `,
    VariadicOf[type_of(T)],
]


@always_inline("builtin")
fn variadic_size[T: type_of(AnyType)](seq: VariadicOf[T]) -> Int:
    return Int(mlir_value=__mlir_op.`pop.variadic.size`(seq))
