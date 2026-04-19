# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct _MLIR:
    comptime KGENParamListType[elt_type: AnyType] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]
    comptime KGENTypeListType[elt_type: type_of(AnyType)] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]


struct Variadic:
    comptime ValuesOfType[type: AnyType] = _MLIR.KGENParamListType[type]
    comptime TypesOfTrait[T: type_of(AnyType)] = _MLIR.KGENTypeListType[T]

    comptime size[T: AnyType, //, seq: Self.ValuesOfType[T]]: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
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
        `#kgen.param_list<>: `, _MLIR.KGENTypeListType[T], `>`
    ]
    comptime types[T: type_of(AnyType), //, *Ts: T] = Ts.values
    comptime values[T: AnyType, //, *elts: T]: Variadic.ValuesOfType[
        T
    ] = elts.values

    # ===-----------------------------------------------------------------------===#
    # VariadicConcat
    # ===-----------------------------------------------------------------------===#

    comptime concat_types[
        T: type_of(AnyType), //, *Ts: Variadic.TypesOfTrait[T]
    ] = __mlir_attr[
        `#kgen.param_list.concat<`, Ts.values, `> :`, Variadic.TypesOfTrait[T]
    ]
    comptime concat_values[
        T: AnyType, //, *Ts: Variadic.ValuesOfType[T]
    ] = __mlir_attr[
        `#kgen.param_list.concat<`, Ts.values, `> :`, Variadic.ValuesOfType[T]
    ]

    # ===-----------------------------------------------------------------------===#
    # Contains
    # ===-----------------------------------------------------------------------===#

    comptime slice_types[
        T: type_of(AnyType),
        //,
        element_types: TypeList[Trait=T, ...],
        start: Int where start >= 0 = 0,
        end: Int where start <= end <= element_types.size = element_types.size,
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[T],
        ParamListType=element_types.values,
        Reducer=_SliceReducer[T, start, end, ...],
    ]


# ===-----------------------------------------------------------------------===#
# VariadicReduce
# ===-----------------------------------------------------------------------===#


comptime _ReduceVariadicIdxGeneratorTypeGenerator[
    Prev: AnyType, From: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": `,
    _MLIR.KGENTypeListType[From],
    `, "Idx":`,
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
    ParamListType: Variadic.TypesOfTrait[From],
    Reducer: _ReduceVariadicIdxGeneratorTypeGenerator[
        Variadic.TypesOfTrait[To], From
    ],
] = __mlir_attr[
    `#kgen.param_list.reduce<`,
    BaseVal,
    `,`,
    ParamListType,
    `,`,
    _IndexToIntWrap[From, Variadic.TypesOfTrait[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]

# ===-----------------------------------------------------------------------===#
# VariadicMap
# ===-----------------------------------------------------------------------===#

comptime _SliceReducer[
    Trait: type_of(AnyType),
    start: Int,
    end: Int,
    Prev: Variadic.TypesOfTrait[Trait],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = (
    Variadic.concat_types[
        Prev, Variadic.types[T=Trait, TypeList[From]()[idx]]
    ] if idx
    >= start
    and idx < end else Prev
)
