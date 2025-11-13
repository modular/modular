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
