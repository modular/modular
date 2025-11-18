# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias VariadicOf[T: type_of(AnyType)] = __mlir_type[`!kgen.variadic<`, T, `>`]

# This specifies a generator to generate a generator type for the mapper of
# [Ts: VariadicOf[AnyType], idx :Int] -> AnyType
alias VariadicIdxToTypeGeneratorTypeGenerator[
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

# This specifies a generator to generate a generator type for the mapper of
# [T: AnyType] -> AnyType
alias TypeToTypeGeneratorTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[`!lit.generator<<"From" :`, From, `>`, To, `>`]

# This is a wrapper that wraps
# generator [v : *Ts, idx : Idx] -> To : {
#    return ToWrap[v[idx]]
# }
alias TypeToTypeWrap[
    From: type_of(AnyType),
    To: type_of(AnyType),
    ToWrap: TypeToTypeGeneratorTypeGenerator[From, To],
    VA: VariadicOf[From],
    idx: __mlir_type.index,
] = ToWrap[VA[idx]]


# Below are two user-facing API to construct the variadic map, we require user
# to specifies the `To` bound because in case of mapping to dependent type, they
# might not have a common bound.

# map(t : variadic[S], mapper : (variadic[S], idx : index) -> D) -> variadic[D]
alias MapVariadicAndIdxToType[
    From: type_of(AnyType), //,
    *,
    To: type_of(AnyType),
    Variadic: __mlir_type[`!kgen.variadic<`, From, `>`],
    Mapper: VariadicIdxToTypeGeneratorTypeGenerator[From, To],
] = __mlir_attr[
    `#kgen.variadic.map<`,
    Variadic,
    `,`,
    Mapper,
    `> : `,
    __mlir_type[`!kgen.variadic<`, To, `>`],
]

# map(t : variadic[S], mapper : (S) -> D) -> variadic[D]
alias MapTypeToType[
    From: type_of(AnyType), //,
    *,
    To: type_of(AnyType),
    Variadic: __mlir_type[`!kgen.variadic<`, From, `>`],
    Mapper: TypeToTypeGeneratorTypeGenerator[From, To],
] = MapVariadicAndIdxToType[
    To=To, Variadic=Variadic, Mapper = TypeToTypeWrap[From, To, Mapper]
]


alias VariadicZip[T: type_of(AnyType), //, *Ts: VariadicOf[T]] = __mlir_attr[
    `#kgen.variadic.zip<`, Ts, `> : !kgen.variadic<`, VariadicOf[T], `>`
]

alias VariadicToTupleMap[
    T: type_of(AnyType), *Ts: VariadicOf[T], idx: __mlir_type.index
] = Tuple[*Ts[idx]]

# This maps a !variadic<!variadic<*elt>> to !variadic<Tuple[*elt]>
# We can not reuse the `MapTypeToType` defined above because this is, technically,
# a `value-to-type map`, since we are mapping a variadic of type values to a tuple.
# We might want to consider to generalized it to a `MapValueToType`, but for now,
# mapping variadic to tuple is the only use case.
alias VariadicToTuple[
    From: type_of(AnyType), //,
    To: type_of(AnyType),
    VariadicToTypeGen: __mlir_type[
        `!lit.generator<<"From": !kgen.variadic<!kgen.variadic<`,
        From,
        `>>, "Idx":`,
        __mlir_type.index,
        `>`,
        To,
        `>`,
    ],
    *Ts: VariadicOf[From],
] = __mlir_attr[
    `#kgen.variadic.map<`,
    Ts,
    `,`,
    VariadicToTypeGen,
    `> : `,
    __mlir_type[`!kgen.variadic<`, To, `>`],
]

# This is all user need to learn.
alias ZipToTuple[T: type_of(AnyType), //, *Ts: VariadicOf[T]] = VariadicToTuple[
    # Give a common bound for all the tuple produced. In test package, we can use AnyType.
    # In stdlib, this should be all the traits that Tuple conforms_to.
    AnyType,
    VariadicToTupleMap[T],
    *VariadicZip[*Ts],  # zipped variadic
]
