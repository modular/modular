# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from builtin._startup import _get_current_or_global_runtime
from memory import UnsafePointer
from memory.unsafe import Pointer
from sys.ffi import RTLD, DLHandle, _get_dylib_function
from pathlib import Path
from utils import StringRef

import _mlir


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    var mof_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", DTypePointer[DType.uint8]
    ](StringRef(".graph_lib"))

    if not mof_lib_path_str_ptr:
        abort("cannot get graph library location from modular.cfg")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getMAXConfigValue` so that it can be destroyed by Mojo.
    var mof_lib_path = String._from_bytes(mof_lib_path_str_ptr)

    if not Path(mof_lib_path).exists():
        abort("cannot load graph library from " + mof_lib_path)

    var ptr = Pointer[DLHandle].alloc(1)
    ptr.store(
        DLHandle(mof_lib_path._strref_dangerous(), RTLD.NOW | RTLD.GLOBAL)
    )
    mof_lib_path._strref_keepalive()
    return ptr.bitcast[NoneType]().address


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn cfunc[func_name: StringLiteral, T: AnyTrivialRegType]() -> T:
    var f = _get_dylib_function[
        "MOF_LIB", func_name, _init_dylib, _destroy_dylib, T
    ]()
    var ptr = Pointer.address_of(f).bitcast[Pointer[NoneType]]().load()
    if not ptr:
        abort("cannot load " + String(func_name) + " from graph library")
    return f


# Note: Keep sections below in sync with capi.mojo, including order, grouping
# and naming.

# Note: Please keep the following naming convention: conept_function. For
# example graph_new, etc.


# ===----------------------------------------------------------------------===#
# Op factories
# ===----------------------------------------------------------------------===#


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
            StringRef,
            _mlir.Type.cType,
        ) -> _mlir.Operation.cType,
    ]()(module.c, loc.c, name._strref_dangerous(), signature.to_mlir().c)


# ===----------------------------------------------------------------------===#
# Attribute factories
# ===----------------------------------------------------------------------===#


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
            StringRef,
            UnsafePointer[T],
            _mlir.Type.cType,
            Bool,
            UnsafePointer[NoneType],
        ) -> _mlir.NamedAttribute.cType,
    ]()(
        name._strref_dangerous(),
        data.data,
        type.c,
        is_owned,
        _get_current_or_global_runtime(),
    )


fn attr_new_tensor(
    name: String,
    data: DTypePointer[DType.invalid],
    type: _mlir.Type,
    is_owned: Bool,
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensor",
        fn (
            StringRef,
            DTypePointer[DType.invalid],
            _mlir.Type.cType,
            Bool,
            UnsafePointer[NoneType],
        ) -> _mlir.NamedAttribute.cType,
    ]()(
        name._strref_dangerous(),
        data,
        type.c,
        is_owned,
        _get_current_or_global_runtime(),
    )


fn attr_new_tensor_from_file(
    name: String, file_name: String, type: _mlir.Type
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensorFromFile",
        fn (
            StringRef, StringRef, _mlir.Type.cType, UnsafePointer[NoneType]
        ) -> _mlir.NamedAttribute.cType,
    ]()(
        name._strref_dangerous(),
        file_name._strref_dangerous(),
        type.c,
        _get_current_or_global_runtime(),
    )


fn attr_new_dim_param_decl(
    ctx: _mlir.Context,
    name: String,
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewDimParamDecl",
        fn (_mlir.Context.cType, StringRef) -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        name._strref_dangerous(),
    )
    return result


fn attr_new_param_decl_array(
    ctx: _mlir.Context,
    params: List[_mlir.Attribute],
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewParamDeclArray",
        fn (
            _mlir.Context.cType,
            Pointer[_mlir.Attribute.cType],
            Int32,
        ) -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        Pointer[_mlir.Attribute](params.data.address).bitcast[
            _mlir.Attribute.cType
        ](),
        len(params),
    )
    return result


# ===----------------------------------------------------------------------===#
# Type helpers
# ===----------------------------------------------------------------------===#


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
            Pointer[_mlir.Attribute.cType],
            Int32,
        ) -> _mlir.Type.cType,
    ]()(
        ctx.c,
        dtype.c,
        ranked,
        Pointer[_mlir.Attribute](dims.data.address).bitcast[
            _mlir.Attribute.cType
        ](),
        len(dims),
    )
    _ = dims
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
        fn (_mlir.Context.cType, StringRef) -> _mlir.Attribute.cType,
    ]()(ctx.c, name._strref_dangerous())


fn dim_is_dynamic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsDynamic", fn (_mlir.Attribute.cType) -> Bool]()(a.c)


fn dim_is_static(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsStatic", fn (_mlir.Attribute.cType) -> Bool]()(a.c)


fn dim_is_symbolic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsSymbolic", fn (_mlir.Attribute.cType) -> Bool]()(
        a.c
    )


fn dim_is_symbolic_expression(a: _mlir.Attribute) -> Bool:
    return cfunc[
        "MAXG_dimIsSymbolicExpression", fn (_mlir.Attribute.cType) -> Bool
    ]()(a.c)


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
