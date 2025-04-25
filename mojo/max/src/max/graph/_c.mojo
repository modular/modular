# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.string import StaticString, StringSlice
from os import abort
from sys.ffi import (
    _get_dylib_function,
    _Global,
    _OwnedDLHandle,
    external_call,
    _find_dylib,
)

import _mlir
from memory import UnsafePointer

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

alias MOF_LIB = _Global["MOF_LIB", _OwnedDLHandle, _init_dylib]


fn _init_dylib() -> _OwnedDLHandle:
    alias key = StaticString(".graph_lib")

    # TODO: Move KGEN_CompilerRT_getMAXConfigValue to a helper somewhere.
    var max_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8]
    ](key.unsafe_ptr(), key.byte_length())

    if not max_lib_path_str_ptr:
        abort("cannot get graph library location from modular.cfg")

    var max_lib_path = String(unsafe_from_utf8_ptr=max_lib_path_str_ptr)
    max_lib_path_str_ptr.free()

    return _find_dylib["graph library"](max_lib_path)


@always_inline
fn cfunc[func_name: StaticString, T: AnyTrivialRegType]() -> T:
    var f = _get_dylib_function[
        MOF_LIB(),
        func_name,
        T,
    ]()
    var ptr = UnsafePointer(to=f).bitcast[UnsafePointer[NoneType]]()[]
    if not ptr:
        abort("cannot load ", func_name, " from graph library")
    return f


# Note: Keep sections below in sync with max_graph.cpp, including order, grouping
# and naming.

# Note: Please keep the following naming convention: conept_function. For
# example graph_new, etc.


# ===-----------------------------------------------------------------------===#
# Op factories
# ===-----------------------------------------------------------------------===#


fn graph_new(
    module: _mlir.Module,
    loc: _mlir.Location,
    name: String,
    signature: _mlir.builtin_types.FunctionType,
) -> _mlir.Operation:
    return cfunc[
        "MAXG_graphNew",
        fn (
            _mlir.Module.cType,
            _mlir.Location.cType,
            StringSlice[__origin_of(name)],
            _mlir.Type.cType,
        ) -> _mlir.Operation.cType,
    ]()(module.c, loc.c, name, signature.to_mlir().c)


# ===-----------------------------------------------------------------------===#
# Attribute factories
# ===-----------------------------------------------------------------------===#


fn attr_new_tensor[
    T: CollectionElement
](
    name: String,
    data: List[T],
    type: _mlir.Type,
    is_owned: Bool,
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensor",
        fn (
            StringSlice[__origin_of(name)],
            UnsafePointer[T],
            _mlir.Type.cType,
            Bool,
        ) -> _mlir.NamedAttribute.cType,
    ]()(name, data.data, type.c, is_owned)


fn attr_new_tensor(
    name: String,
    data: UnsafePointer[NoneType],
    type: _mlir.Type,
    is_owned: Bool,
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensor",
        fn (
            StringSlice[__origin_of(name)],
            UnsafePointer[NoneType],
            _mlir.Type.cType,
            Bool,
        ) -> _mlir.NamedAttribute.cType,
    ]()(name, data, type.c, is_owned)


fn attr_new_tensor_from_file(
    name: String, file_name: String, type: _mlir.Type
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensorFromFile",
        fn (
            StringSlice[__origin_of(name)],
            StringSlice[__origin_of(file_name)],
            _mlir.Type.cType,
        ) -> _mlir.NamedAttribute.cType,
    ]()(name, file_name, type.c)


fn attr_new_dim_param_decl(
    ctx: _mlir.Context,
    name: String,
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewDimParamDecl",
        fn (
            _mlir.Context.cType, StringSlice[__origin_of(name)]
        ) -> _mlir.Attribute.cType,
    ]()(ctx.c, name)
    return result


fn attr_new_param_decl_array(
    ctx: _mlir.Context,
    params: List[_mlir.Attribute],
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewParamDeclArray",
        fn (
            _mlir.Context.cType,
            UnsafePointer[_mlir.Attribute.cType],
            Int32,
        ) -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        UnsafePointer[_mlir.Attribute](params.data).bitcast[
            _mlir.Attribute.cType
        ](),
        len(params),
    )
    return result


fn attr_new_shape(
    ctx: _mlir.Context,
    dims: List[_mlir.Attribute],
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewShape",
        fn (
            _mlir.Context.cType,
            UnsafePointer[_mlir.Attribute.cType],
            Int32,
        ) -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        UnsafePointer[_mlir.Attribute](dims.data).bitcast[
            _mlir.Attribute.cType
        ](),
        len(dims),
    )
    return result


# ===-----------------------------------------------------------------------===#
# Type helpers
# ===-----------------------------------------------------------------------===#


fn dtype_new(ctx: _mlir.Context, dtype: DType) -> _mlir.Type:
    return cfunc[
        "MAXG_dTypeNew", fn (_mlir.Context.cType, UInt8) -> _mlir.Type.cType
    ]()(ctx.c, dtype._as_i8())


fn dim_type_new_dynamic() -> Int64:
    return cfunc["MAXG_dimTypeNewDynamic", fn () -> Int64]()()


fn tensor_type_new(
    ctx: _mlir.Context,
    dtype: _mlir.Type,
    dims: List[_mlir.Attribute],
    ranked: Bool,
) -> _mlir.Type:
    var result = cfunc[
        "MAXG_tensorTypeNew",
        fn (
            _mlir.Context.cType,
            _mlir.Type.cType,
            Bool,
            UnsafePointer[_mlir.Attribute.cType],
            Int32,
        ) -> _mlir.Type.cType,
    ]()(
        ctx.c,
        dtype.c,
        ranked,
        UnsafePointer[_mlir.Attribute](dims.data).bitcast[
            _mlir.Attribute.cType
        ](),
        len(dims),
    )
    return result


fn tensor_type_get_dtype(v: _mlir.Type) -> DType:
    var dtype = cfunc[
        "MAXG_tensorTypeGetDType", fn (_mlir.Type.cType) -> UInt8
    ]()(v.c)
    return DType._from_ui8(dtype.value)


fn tensor_type_is_ranked(v: _mlir.Type) -> Bool:
    return cfunc["MAXG_tensorTypeIsRanked", fn (_mlir.Type.cType) -> Bool]()(
        v.c
    )


fn tensor_type_get_rank(t: _mlir.Type) -> Int64:
    return cfunc["MAXG_tensorTypeGetRank", fn (_mlir.Type.cType) -> Int64]()(
        t.c
    )


fn tensor_type_get_dim(t: _mlir.Type, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_tensorTypeShapeGetDim",
        fn (_mlir.Type.cType, Int64) -> _mlir.Attribute.cType,
    ]()(t.c, dim)


fn dim_new_dynamic(ctx: _mlir.Context) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewDynamic",
        fn (_mlir.Context.cType) -> _mlir.Attribute.cType,
    ]()(ctx.c)


fn dim_new_static(ctx: _mlir.Context, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewStatic",
        fn (_mlir.Context.cType, Int64) -> _mlir.Attribute.cType,
    ]()(ctx.c, dim)


fn dim_new_symbolic(ctx: _mlir.Context, name: String) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewSymbolic",
        fn (
            _mlir.Context.cType, StringSlice[__origin_of(name)]
        ) -> _mlir.Attribute.cType,
    ]()(ctx.c, name)


fn dim_is_dynamic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsDynamic", fn (_mlir.Attribute.cType) -> Bool]()(a.c)


fn dim_is_static(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsStatic", fn (_mlir.Attribute.cType) -> Bool]()(a.c)


fn dim_is_symbolic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsSymbolic", fn (_mlir.Attribute.cType) -> Bool]()(
        a.c
    )


fn dim_is_algebraic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsAlgebraic", fn (_mlir.Attribute.cType) -> Bool]()(
        a.c
    )


fn dim_static_value(a: _mlir.Attribute) -> Int64:
    return cfunc["MAXG_dimStaticValue", fn (_mlir.Attribute.cType) -> Int64]()(
        a.c
    )


fn dim_symbolic_name(a: _mlir.Attribute) -> _mlir.Identifier:
    return cfunc[
        "MAXG_dimSymbolicName",
        fn (_mlir.Attribute.cType) -> _mlir.Identifier.cType,
    ]()(a.c)


fn list_type_new(ctx: _mlir.Context, eltype: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeNew",
        fn (_mlir.Context.cType, _mlir.Type.cType) -> _mlir.Type.cType,
    ]()(ctx.c, eltype.c)


fn list_type_element_type(t: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeElementType", fn (_mlir.Type.cType) -> _mlir.Type.cType
    ]()(t.c)


fn type_is_list(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsList", fn (_mlir.Type.cType) -> Bool]()(t.c)


fn type_is_tensor(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsTensor", fn (_mlir.Type.cType) -> Bool]()(t.c)


fn type_is_opaque(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsOpaque", fn (_mlir.Type.cType) -> Bool]()(t.c)


fn opaque_type_new(ctx: _mlir.Context, name: String) -> _mlir.Type:
    return cfunc[
        "MAXG_opaqueTypeNew",
        fn (
            _mlir.Context.cType, StringSlice[__origin_of(name)]
        ) -> _mlir.Type.cType,
    ]()(ctx.c, name)


fn opaque_type_name(t: _mlir.Type) -> StaticString:
    return cfunc[
        "MAXG_opaqueTypeName", fn (_mlir.Type.cType) -> StaticString
    ]()(t.c)
