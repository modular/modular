# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from tensor import Tensor
from sys.ffi import RTLD, DLHandle, _get_dylib_function
from pathlib import Path
from utils import StringRef

import _mlir


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    var mof_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getConfigValue", DTypePointer[DType.int8]
    ]("max.graph_lib")

    if not mof_lib_path_str_ptr:
        abort("cannot get graph library location from modular.cfg")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getConfigValue` so that it can be destroyed by Mojo.
    var mof_lib_path = String._from_bytes(mof_lib_path_str_ptr)

    if not Path(mof_lib_path).exists():
        abort("cannot load graph library from " + mof_lib_path)

    var ptr = Pointer[DLHandle].alloc(1)
    ptr.store(
        DLHandle(mof_lib_path._strref_dangerous(), RTLD.NOW | RTLD.GLOBAL)
    )
    mof_lib_path._strref_keepalive()
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn cfunc[func_name: StringLiteral, T: AnyRegType]() -> T:
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
            _mlir.Module.c_type,
            _mlir.Location.c_type,
            StringRef,
            _mlir.Type.c_type,
        ) -> _mlir.Operation.c_type,
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
            AnyPointer[T],
            _mlir.Type.c_type,
            Bool,
        ) -> _mlir.NamedAttribute.c_type,
    ]()(name._strref_dangerous(), data.data, type.c, is_owned)


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
            _mlir.Type.c_type,
            Bool,
        ) -> _mlir.NamedAttribute.c_type,
    ]()(name._strref_dangerous(), data, type.c, is_owned)


fn attr_new_tensor_from_file(
    name: String, file_name: String, type: _mlir.Type
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensorFromFile",
        fn (
            StringRef, StringRef, _mlir.Type.c_type
        ) -> _mlir.NamedAttribute.c_type,
    ]()(name._strref_dangerous(), file_name._strref_dangerous(), type.c)


# ===----------------------------------------------------------------------===#
# Type helpers
# ===----------------------------------------------------------------------===#


fn dtype_new(ctx: _mlir.Context, dtype: DType) -> _mlir.Type:
    return cfunc[
        "MAXG_dTypeNew", fn (_mlir.Context.c_type, UInt8) -> _mlir.Type.c_type
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
            _mlir.Context.c_type,
            _mlir.Type.c_type,
            Bool,
            Pointer[_mlir.Attribute.c_type],
            Int32,
        ) -> _mlir.Type.c_type,
    ]()(
        ctx.c,
        dtype.c,
        ranked,
        Pointer[_mlir.Attribute](dims.data.value).bitcast[
            _mlir.Attribute.c_type
        ](),
        len(dims),
    )
    _ = dims
    return result


fn tensor_type_get_dtype(v: _mlir.Type) -> DType:
    var dtype = cfunc[
        "MAXG_tensorTypeGetDType", fn (_mlir.Type.c_type) -> UInt8
    ]()(v.c)
    return DType._from_ui8(dtype.value)


fn tensor_type_is_ranked(v: _mlir.Type) -> Bool:
    return cfunc["MAXG_tensorTypeIsRanked", fn (_mlir.Type.c_type) -> Bool]()(
        v.c
    )


fn tensor_type_get_rank(t: _mlir.Type) -> Int64:
    return cfunc["MAXG_tensorTypeGetRank", fn (_mlir.Type.c_type) -> Int64]()(
        t.c
    )


fn tensor_type_get_dim(t: _mlir.Type, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_tensorTypeShapeGetDim",
        fn (_mlir.Type.c_type, Int64) -> _mlir.Attribute.c_type,
    ]()(t.c, dim)


fn dim_new_dynamic(ctx: _mlir.Context) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewDynamic",
        fn (_mlir.Context.c_type) -> _mlir.Attribute.c_type,
    ]()(ctx.c)


fn dim_new_static(ctx: _mlir.Context, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewStatic",
        fn (_mlir.Context.c_type, Int64) -> _mlir.Attribute.c_type,
    ]()(ctx.c, dim)


fn dim_new_symbolic(ctx: _mlir.Context, name: String) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewSymbolic",
        fn (_mlir.Context.c_type, StringRef) -> _mlir.Attribute.c_type,
    ]()(ctx.c, name._strref_dangerous())


fn dim_is_dynamic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsDynamic", fn (_mlir.Attribute.c_type) -> Bool]()(
        a.c
    )


fn dim_is_static(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsStatic", fn (_mlir.Attribute.c_type) -> Bool]()(a.c)


fn dim_is_symbolic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsSymbolic", fn (_mlir.Attribute.c_type) -> Bool]()(
        a.c
    )


fn dim_is_symbolic_expression(a: _mlir.Attribute) -> Bool:
    return cfunc[
        "MAXG_dimIsSymbolicExpression", fn (_mlir.Attribute.c_type) -> Bool
    ]()(a.c)


fn dim_static_value(a: _mlir.Attribute) -> Int64:
    return cfunc["MAXG_dimStaticValue", fn (_mlir.Attribute.c_type) -> Int64]()(
        a.c
    )


fn dim_symbolic_name(a: _mlir.Attribute) -> _mlir.Identifier:
    return cfunc[
        "MAXG_dimSymbolicName",
        fn (_mlir.Attribute.c_type) -> _mlir.Identifier.c_type,
    ]()(a.c)


fn list_type_new(ctx: _mlir.Context, eltype: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeNew",
        fn (_mlir.Context.c_type, _mlir.Type.c_type) -> _mlir.Type.c_type,
    ]()(ctx.c, eltype.c)


fn list_type_element_type(t: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeElementType", fn (_mlir.Type.c_type) -> _mlir.Type.c_type
    ]()(t.c)


fn type_is_list(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsList", fn (_mlir.Type.c_type) -> Bool]()(t.c)


fn type_is_tensor(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsTensor", fn (_mlir.Type.c_type) -> Bool]()(t.c)
