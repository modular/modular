# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Variadic:
    comptime ValuesOfType[type: AnyType] = __mlir_type[
        `!kgen.variadic<`, type, `>`
    ]
    comptime TypesOfTrait[T: type_of(AnyType)] = __mlir_type[
        `!kgen.variadic<`, T, `>`
    ]

    comptime size[T: AnyType, //, seq: Self.ValuesOfType[T]]: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.variadic.size<:`,
            type_of(seq),
            ` `,
            +seq,
            `> : index`,
        ]
    )

    comptime size_types[
        T: type_of(AnyType), //, seq: Self.TypesOfTrait[T]
    ]: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.variadic.size<:`,
            type_of(seq),
            ` `,
            +seq,
            `> : index`,
        ]
    )

    # ===-----------------------------------------------------------------------===#
    # Utils
    # ===-----------------------------------------------------------------------===#

    comptime empty_of_trait[T: type_of(AnyType)] = __mlir_attr[
        `#kgen.variadic<>: !kgen.variadic<`, T, `>`
    ]
    comptime empty_of_type[T: AnyType] = __mlir_attr[
        `#kgen.variadic<>: !kgen.variadic<`, T, `>`
    ]
    comptime types[T: type_of(AnyType), //, *Ts: T] = Ts
    comptime values[T: AnyType, //, *values_: T]: Variadic.ValuesOfType[
        T
    ] = values_

    # ===-----------------------------------------------------------------------===#
    # VariadicConcat
    # ===-----------------------------------------------------------------------===#

    comptime concat_types[
        T: type_of(AnyType), //, *Ts: Variadic.TypesOfTrait[T]
    ] = __mlir_attr[
        `#kgen.variadic.concat<`, Ts, `> :`, Variadic.TypesOfTrait[T]
    ]
    comptime concat_values[
        T: AnyType, //, *Ts: Variadic.ValuesOfType[T]
    ] = __mlir_attr[
        `#kgen.variadic.concat<`, Ts, `> :`, Variadic.ValuesOfType[T]
    ]
    comptime reverse[
        T: type_of(AnyType), //, *element_types: T
    ] = _MapVariadicAndIdxToType[
        To=T, VariadicType=element_types, Mapper=_ReversedVariadic[T, ...]
    ]
    comptime splat_type[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ]: Variadic.TypesOfTrait[Trait] = Self.tabulate_type[
        Trait=Trait, ToT=type, count, _SplatTypeTabulator[Trait, type, _]
    ]
    comptime splat_value[
        T: AnyType, //, count: Int, value: T
    ]: Variadic.ValuesOfType[T] = Self.tabulate[
        count, _SplatValueTabulator[value, _]
    ]

    # ===-----------------------------------------------------------------------===#
    # Tabulate
    # ===-----------------------------------------------------------------------===#

    # tabulate: Apply an "index -> value" generator, N times to build a variadic.
    comptime tabulate[
        ToT: AnyType,
        //,
        count: Int,
        Mapper: _TabulateIntToValueGeneratorType[ToT],
    ]: Variadic.ValuesOfType[ToT] = __mlir_attr[
        `#kgen.variadic.tabulate<`,
        count._mlir_value,
        `,`,
        _IndexToIntTabulateWrap[Mapper, ...],
        `> : `,
        Variadic.ValuesOfType[ToT],
    ]

    comptime tabulate_type[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        count: Int,
        Mapper: _TabulateIntToTypeGeneratorType[Trait, ToT],
    ]: Variadic.TypesOfTrait[Trait] = __mlir_attr[
        `#kgen.variadic.tabulate<`,
        count._mlir_value,
        `,`,
        _IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
        `> : `,
        Variadic.TypesOfTrait[Trait],
    ]

    # ===-----------------------------------------------------------------------===#
    # Contains
    # ===-----------------------------------------------------------------------===#

    comptime contains[
        Trait: type_of(AnyType),
        //,
        type: Trait,
        element_types: Variadic.TypesOfTrait[Trait],
    ] = _ReduceVariadicAndIdxToValue[
        BaseVal=Variadic.values[False],
        VariadicType=element_types,
        #  Curry `_ContainsMapper` to fit the reducer signature
        Reducer=_ContainsReducer[Trait=Trait, Type=type, ...],
    ][
        0
    ]
    comptime map_types_to_types[
        From: type_of(AnyType),
        To: type_of(AnyType),
        //,
        element_types: Variadic.TypesOfTrait[From],
        Mapper: _TypeToTypeGenerator[From, To],
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[To],
        VariadicType=element_types,
        Reducer=_MapTypeToTypeReducer[From, To, Mapper, ...],
    ]
    comptime slice_types[
        T: type_of(AnyType),
        //,
        element_types: Variadic.TypesOfTrait[T],
        start: Int where start >= 0 = 0,
        end: Int where (
            start <= end <= Variadic.size_types[element_types]
        ) = Variadic.size_types[element_types],
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[T],
        VariadicType=element_types,
        Reducer=_SliceReducer[T, start, end, ...],
    ]
    comptime zip_types[
        Trait: type_of(AnyType), //, *types: Variadic.TypesOfTrait[Trait]
    ] = __mlir_attr[
        `#kgen.variadic.zip<`,
        types,
        `> : !kgen.variadic<`,
        Variadic.TypesOfTrait[Trait],
        `>`,
    ]
    comptime zip_values[
        type: AnyType, //, *values: Variadic.ValuesOfType[type]
    ] = __mlir_attr[
        `#kgen.variadic.zip<`,
        values,
        `> : !kgen.variadic<`,
        Variadic.ValuesOfType[type],
        `>`,
    ]
    comptime filter_types[
        T: type_of(AnyType),
        //,
        *element_types: T,
        predicate: _TypePredicateGenerator[T],
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[T],
        VariadicType=element_types,
        Reducer=_FilterReducer[T, predicate, ...],
    ]
    comptime _ValueIdxToValueGeneratorType[
        From: AnyType, To: AnyType
    ] = __mlir_type[
        `!lit.generator<<"From": `,
        +From,
        `, "Idx":`,
        Int,
        `>`,
        +To,
        `>`,
    ]
    comptime _ValueToValueMapper[
        FromType: AnyType,
        ToType: AnyType,
        //,
        Mapper: Variadic._ValueIdxToValueGeneratorType[FromType, ToType],
        Prev: Variadic.ValuesOfType[ToType],
        From: Variadic.ValuesOfType[FromType],
        idx: Int,
    ] = Variadic.concat_values[
        Prev,
        Variadic.values[Mapper[From[idx], idx]],
    ]


# ===-----------------------------------------------------------------------===#
# Tabulate Helpers
# ===-----------------------------------------------------------------------===#


comptime _TabulateIntToValueGeneratorType[ToT: AnyType] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `>`,
    +ToT,
    `>`,
]

comptime _TabulateIntToTypeGeneratorType[
    Trait: type_of(AnyType), ToT: Trait
] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `> `,
    Trait,
    `>`,
]


comptime _IndexToIntTabulateWrap[
    ToT: AnyType,
    //,
    ToWrap: _TabulateIntToValueGeneratorType[ToT],
    idx: __mlir_type.index,
]: ToT = ToWrap[Int(mlir_value=idx)]

comptime _IndexToIntTypeTabulateWrap[
    Trait: type_of(AnyType),
    ToT: Trait,
    //,
    ToWrap: _TabulateIntToTypeGeneratorType[Trait, ToT],
    idx: __mlir_type.index,
] = ToWrap[Int(mlir_value=idx)]


comptime _SplatValueTabulator[T: AnyType, //, value: T, index: Int] = value
comptime _SplatTypeTabulator[
    Trait: type_of(AnyType), T: Trait, index: Int
]: Trait = T


# ===-----------------------------------------------------------------------===#
# VariadicReduce
# ===-----------------------------------------------------------------------===#


comptime _ReduceVariadicIdxGeneratorTypeGenerator[
    Prev: AnyType, From: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    +Prev,
    `>`,
]
comptime _IndexToIntWrap[
    From: type_of(AnyType),
    ReduceT: AnyType,
    ToWrap: _ReduceVariadicIdxGeneratorTypeGenerator[ReduceT, From],
    PrevV: ReduceT,
    VA: Variadic.TypesOfTrait[From],
    idx: __mlir_type.index,
] = ToWrap[PrevV, VA, Int(mlir_value=idx)]

comptime _ReduceVariadicAndIdxToVariadic[
    From: type_of(AnyType),
    To: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.TypesOfTrait[To],
    VariadicType: Variadic.TypesOfTrait[From],
    Reducer: _ReduceVariadicIdxGeneratorTypeGenerator[
        Variadic.TypesOfTrait[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntWrap[From, Variadic.TypesOfTrait[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
comptime _ReduceValueIdxGeneratorTypeGenerator[
    Prev: AnyType, From: AnyType
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    +Prev,
    `>`,
]
comptime _IndexToIntValueWrap[
    From: AnyType,
    ReduceT: AnyType,
    ToWrap: _ReduceValueIdxGeneratorTypeGenerator[ReduceT, From],
    PrevV: ReduceT,
    VA: Variadic.ValuesOfType[From],
    idx: __mlir_type.index,
] = ToWrap[PrevV, VA, Int(mlir_value=idx)]


comptime _ReduceValueAndIdxToVariadic[
    From: AnyType,
    To: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.TypesOfTrait[To],
    VariadicType: Variadic.ValuesOfType[From],
    Reducer: _ReduceValueIdxGeneratorTypeGenerator[
        Variadic.TypesOfTrait[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntValueWrap[From, Variadic.TypesOfTrait[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
comptime _ReduceVariadicAndIdxToValue[
    To: AnyType,
    From: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.ValuesOfType[To],
    VariadicType: Variadic.TypesOfTrait[From],
    Reducer: _ReduceVariadicIdxGeneratorTypeGenerator[
        Variadic.ValuesOfType[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntWrap[From, Variadic.ValuesOfType[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
# ===-----------------------------------------------------------------------===#
# VariadicMap
# ===-----------------------------------------------------------------------===#

comptime _TypeToTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[`!lit.generator<<"From":`, From, `>`, To, `>`]
comptime _VariadicIdxToTypeGeneratorTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    To,
    `>`,
]
comptime _WrapVariadicIdxToTypeMapperToReducer[
    F: type_of(AnyType),
    T: type_of(AnyType),
    Mapper: _VariadicIdxToTypeGeneratorTypeGenerator[F, T],
    Prev: Variadic.TypesOfTrait[T],
    From: Variadic.TypesOfTrait[F],
    Idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[Mapper[From, Idx]]]


comptime _MapVariadicAndIdxToType[
    From: type_of(AnyType),
    //,
    *,
    To: type_of(AnyType),
    VariadicType: Variadic.TypesOfTrait[From],
    Mapper: _VariadicIdxToTypeGeneratorTypeGenerator[From, To],
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal=Variadic.empty_of_trait[To],  # reduce from a empty variadic
    VariadicType=VariadicType,
    Reducer=_WrapVariadicIdxToTypeMapperToReducer[From, To, Mapper, ...],
]
comptime MapVariadicAndIdxToType = _MapVariadicAndIdxToType
comptime _VariadicValuesIdxToTypeGeneratorTypeGenerator[
    From: AnyType, To: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    To,
    `>`,
]
comptime _WrapVariadicValuesIdxToTypeMapperToReducer[
    F: AnyType,
    T: type_of(AnyType),
    Mapper: _VariadicValuesIdxToTypeGeneratorTypeGenerator[F, T],
    Prev: Variadic.TypesOfTrait[T],
    From: Variadic.ValuesOfType[F],
    Idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[Mapper[From, Idx]]]

comptime _ReversedVariadic[
    T: type_of(AnyType),
    element_types: Variadic.TypesOfTrait[T],
    idx: Int,
] = element_types[Variadic.size_types[element_types] - 1 - idx]
comptime _ContainsReducer[
    Trait: type_of(AnyType),
    Type: Trait,
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = Variadic.values[_type_is_eq_parse_time[From[idx], Type]() or Prev[0]]

comptime _MapTypeToTypeReducer[
    FromTrait: type_of(AnyType),
    ToTrait: type_of(AnyType),
    Mapper: _TypeToTypeGenerator[FromTrait, ToTrait],
    Prev: Variadic.TypesOfTrait[ToTrait],
    From: Variadic.TypesOfTrait[FromTrait],
    idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[T=ToTrait, Mapper[From[idx]]]]

comptime _SliceReducer[
    Trait: type_of(AnyType),
    start: Int,
    end: Int,
    Prev: Variadic.TypesOfTrait[Trait],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = (
    Variadic.concat_types[Prev, Variadic.types[T=Trait, From[idx]]] if idx
    >= start
    and idx < end else Prev
)
comptime _TypePredicateGenerator[T: type_of(AnyType)] = __mlir_type[
    `!lit.generator<<"Type": `,
    T,
    `>`,
    Bool,
    `>`,
]
comptime _FilterReducer[
    Trait: type_of(AnyType),
    Predicate: _TypePredicateGenerator[Trait],
    Prev: Variadic.TypesOfTrait[Trait],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = (
    Variadic.concat_types[
        Prev, Variadic.types[T=Trait, From[idx]]
    ] if Predicate[From[idx]] else Prev
)
