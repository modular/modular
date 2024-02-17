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

import mlir


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


fn _init_dylib(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    let mof_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getConfigValue", DTypePointer[DType.int8]
    ]("max.graph_lib")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getConfigValue` so that it can be destroyed by Mojo.
    let pathlen = len(StringRef(mof_lib_path_str_ptr))
    let mof_lib_path = String(
        mof_lib_path_str_ptr, pathlen + 1
    )  # account for the terminator

    if not mof_lib_path:
        print("cannot get graph library location from modular.cfg")
        trap()

    if not Path(mof_lib_path).exists():
        print("cannot load graph library from " + mof_lib_path)
        trap()

    let ptr = Pointer[DLHandle].alloc(1)
    ptr.store(
        DLHandle(mof_lib_path._strref_dangerous(), RTLD.NOW | RTLD.GLOBAL)
    )
    mof_lib_path._strref_keepalive()
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: Pointer[NoneType]):
    __get_address_as_lvalue(ptr.bitcast[DLHandle]().address)._del_old()
    ptr.free()


@always_inline
fn cfunc[T: AnyRegType](name: StringRef) -> T:
    var f = _get_dylib_function["MOF_LIB", _init_dylib, _destroy_dylib, T](name)
    let ptr = Pointer.address_of(f).bitcast[Pointer[NoneType]]().load()
    if not ptr:
        print("cannot load " + String(name) + " from graph library")
        trap()
    return f


# Note: Keep sections below in sync with capi.mojo, including order, grouping
# and naming.

# Note: Please keep the following naming convention: conept_function. For
# example graph_new, etc.


# ===----------------------------------------------------------------------===#
# Op factories
# ===----------------------------------------------------------------------===#


fn graph_new(
    module: mlir.Module,
    loc: mlir.Location,
    name: StringRef,
    signature: mlir.builtin_types.FunctionType,
) -> mlir.Operation:
    return cfunc[
        fn (
            mlir.Module.c_type,
            mlir.Location.c_type,
            StringRef,
            mlir.Type.c_type,
        ) -> mlir.Operation.c_type
    ]("MAXG_graphNew")(module.c, loc.c, name, signature.to_mlir().c)


# ===----------------------------------------------------------------------===#
# Attribute factories
# ===----------------------------------------------------------------------===#


fn attr_new_tensor[
    T: CollectionElement
](
    m: mlir.Module,
    name: StringRef,
    data: DynamicVector[T],
    type: mlir.Type,
    is_owned: Bool,
) -> mlir.NamedAttribute:
    return cfunc[
        fn (
            mlir.Module.c_type, StringRef, AnyPointer[T], mlir.Type.c_type, Bool
        ) -> mlir.NamedAttribute.c_type
    ]("MAXG_attrNewTensor")(m.c, name, data.data, type.c, is_owned)


fn attr_new_tensor(
    m: mlir.Module,
    name: StringRef,
    data: DTypePointer[DType.invalid],
    type: mlir.Type,
    is_owned: Bool,
) -> mlir.NamedAttribute:
    return cfunc[
        fn (
            mlir.Module.c_type,
            StringRef,
            DTypePointer[DType.invalid],
            mlir.Type.c_type,
            Bool,
        ) -> mlir.NamedAttribute.c_type
    ]("MAXG_attrNewTensor")(m.c, name, data, type.c, is_owned)


fn attr_new_tensor_from_file(
    m: mlir.Module, name: StringRef, file_name: StringRef, type: mlir.Type
) -> mlir.NamedAttribute:
    return cfunc[
        fn (
            mlir.Module.c_type, StringRef, StringRef, mlir.Type.c_type
        ) -> mlir.NamedAttribute.c_type
    ]("MAXG_attrNewTensorFromFile")(m.c, name, file_name, type.c)


# ===----------------------------------------------------------------------===#
# Type helpers
# ===----------------------------------------------------------------------===#


fn dtype_new(m: mlir.Module, dtype: DType) -> mlir.Type:
    return cfunc[fn (mlir.Module.c_type, UInt8) -> mlir.Type.c_type](
        "MAXG_dTypeNew"
    )(m.c, dtype._as_i8())


fn dim_type_new_dynamic() -> Int64:
    return cfunc[fn () -> Int64]("MAXG_dimTypeNewDynamic")()


fn tensor_type_new(
    m: mlir.Module,
    dtype: mlir.Type,
    dims: DynamicVector[mlir.Attribute],
    ranked: Bool,
) -> mlir.Type:
    let result = cfunc[
        fn (
            mlir.Module.c_type,
            mlir.Type.c_type,
            Bool,
            Pointer[mlir.Attribute.c_type],
            Int32,
        ) -> mlir.Type.c_type
    ]("MAXG_tensorTypeNew")(
        m.c,
        dtype.c,
        ranked,
        Pointer[mlir.Attribute](dims.data.value).bitcast[
            mlir.Attribute.c_type
        ](),
        len(dims),
    )
    _ = dims
    return result


fn tensor_type_get_dtype(v: mlir.Type) -> DType:
    let dtype = cfunc[fn (mlir.Type.c_type) -> UInt8](
        "MAXG_tensorTypeGetDType"
    )(v.c)
    return DType._from_ui8(dtype.value)


fn tensor_type_is_ranked(v: mlir.Type) -> Bool:
    return cfunc[fn (mlir.Type.c_type) -> Bool]("MAXG_tensorTypeIsRanked")(v.c)


fn tensor_type_get_rank(t: mlir.Type) -> Int64:
    return cfunc[fn (mlir.Type.c_type) -> Int64]("MAXG_tensorTypeGetRank")(t.c)


fn tensor_type_get_dim(t: mlir.Type, dim: Int64) -> mlir.Attribute:
    return cfunc[fn (mlir.Type.c_type, Int64) -> mlir.Attribute.c_type](
        "MAXG_tensorTypeShapeGetDim"
    )(t.c, dim)


fn dim_new_dynamic(ctx: mlir.Context) -> mlir.Attribute:
    return cfunc[fn (mlir.Context.c_type) -> mlir.Attribute.c_type](
        "MAXG_dimNewDynamic"
    )(ctx.c)


fn dim_new_static(ctx: mlir.Context, dim: Int64) -> mlir.Attribute:
    return cfunc[fn (mlir.Context.c_type, Int64) -> mlir.Attribute.c_type](
        "MAXG_dimNewStatic"
    )(ctx.c, dim)


fn dim_new_symbolic(ctx: mlir.Context, name: StringRef) -> mlir.Attribute:
    return cfunc[fn (mlir.Context.c_type, StringRef) -> mlir.Attribute.c_type](
        "MAXG_dimNewSymbolic"
    )(ctx.c, name)


fn dim_is_dynamic(a: mlir.Attribute) -> Bool:
    return cfunc[fn (mlir.Attribute.c_type) -> Bool]("MAXG_dimIsDynamic")(a.c)


fn dim_is_static(a: mlir.Attribute) -> Bool:
    return cfunc[fn (mlir.Attribute.c_type) -> Bool]("MAXG_dimIsStatic")(a.c)


fn dim_is_symbolic(a: mlir.Attribute) -> Bool:
    return cfunc[fn (mlir.Attribute.c_type) -> Bool]("MAXG_dimIsSymbolic")(a.c)


fn dim_is_symbolic_expression(a: mlir.Attribute) -> Bool:
    return cfunc[fn (mlir.Attribute.c_type) -> Bool](
        "MAXG_dimIsSymbolicExpression"
    )(a.c)


fn dim_static_value(a: mlir.Attribute) -> Int64:
    return cfunc[fn (mlir.Attribute.c_type) -> Int64]("MAXG_dimStaticValue")(
        a.c
    )


fn dim_symbolic_name(a: mlir.Attribute) -> mlir.Identifier:
    return cfunc[fn (mlir.Attribute.c_type) -> mlir.Identifier.c_type](
        "MAXG_dimSymbolicName"
    )(a.c)


fn list_type_new(m: mlir.Module, eltype: mlir.Type) -> mlir.Type:
    return cfunc[fn (mlir.Module.c_type, mlir.Type.c_type) -> mlir.Type.c_type](
        "MAXG_listTypeNew"
    )(m.c, eltype.c)


fn list_type_element_type(t: mlir.Type) -> mlir.Type:
    return cfunc[fn (mlir.Type.c_type) -> mlir.Type.c_type](
        "MAXG_listTypeElementType"
    )(t.c)


fn type_is_list(t: mlir.Type) -> Bool:
    return cfunc[fn (mlir.Type.c_type) -> Bool]("MAXG_typeIsList")(t.c)


fn type_is_tensor(t: mlir.Type) -> Bool:
    return cfunc[fn (mlir.Type.c_type) -> Bool]("MAXG_typeIsTensor")(t.c)
