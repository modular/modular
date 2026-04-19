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
