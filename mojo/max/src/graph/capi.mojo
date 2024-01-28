# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from tensor import Tensor
from sys.ffi import RTLD, DLHandle, _get_dylib_function

import mlir


struct _AttrMap:
    """Opaque data type mapping to M_AttrMap in the Builder C API."""

    pass


struct _Graph:
    """Opaque data type mapping to M_Graph in the Builder C API."""

    pass


struct _Tuple:
    """Opaque data type mapping to M_Tuple in the Builder C API."""

    pass


struct _Symbol:
    """Opaque data type mapping to M_Symbol in the Builder C API."""

    pass


struct _Arity:
    """Opaque data type mapping to M_Arity in the Builder C API."""

    pass


struct _TensorTuple:
    """Opaque data type mapping to M_TensorTuple in the MOF C API."""

    pass


alias AttrMapPtr = Pointer[_AttrMap]
alias GraphPtr = Pointer[_Graph]
alias TuplePtr = Pointer[_Tuple]
alias SymbolPtr = Pointer[_Symbol]
alias ArityPtr = Pointer[_Arity]
alias TensorTuplePtr = Pointer[_TensorTuple]


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


fn attr_new_string(
    m: mlir.Module, name: StringRef, value: StringRef
) -> mlir.NamedAttribute:
    return cfunc[
        fn (
            mlir.Module.c_type, StringRef, StringRef
        ) -> mlir.NamedAttribute.c_type
    ]("MAXG_attrNewString")(m.c, name, value)


# ===----------------------------------------------------------------------===#
# AttrMap support
# ===----------------------------------------------------------------------===#


fn attr_map_new() -> AttrMapPtr:
    return cfunc[fn () -> AttrMapPtr]("MAXG_attrMapNew")()


fn attr_map_add_attr(m: AttrMapPtr, a: mlir.NamedAttribute):
    return cfunc[fn (AttrMapPtr, mlir.NamedAttribute.c_type) -> NoneType](
        "MAXG_attrMapAddAttr"
    )(m, a.c())


fn attr_map_size(m: AttrMapPtr) -> Int:
    return cfunc[fn (AttrMapPtr) -> Int]("MAXG_attrMapSize")(m)


# ===----------------------------------------------------------------------===#
# Graph support
# ===----------------------------------------------------------------------===#


fn graph_new(
    m: mlir.Module,
    loc: mlir.Location,
    name: StringRef,
    in_types: ArityPtr,
    out_types: ArityPtr,
) -> GraphPtr:
    return cfunc[
        fn (
            mlir.Module.c_type,
            mlir.Location.c_type,
            StringRef,
            ArityPtr,
            ArityPtr,
        ) -> GraphPtr
    ]("MAXG_graphNew")(m.c, loc.c, name, in_types, out_types)


fn graph_get_module(g: GraphPtr) -> mlir.Module:
    return cfunc[fn (GraphPtr) -> mlir.Module.c_type]("MAXG_graphGetModule")(g)


fn graph_get_arg(g: GraphPtr, pos: UInt32) -> SymbolPtr:
    return cfunc[fn (GraphPtr, UInt32) -> SymbolPtr]("MAXG_graphGetArg")(g, pos)


fn graph_new_op(
    g: GraphPtr,
    loc: mlir.Location,
    name: StringRef,
    inputs: TuplePtr,
    out_types: ArityPtr,
    attrs: AttrMapPtr,
) -> TuplePtr:
    return cfunc[
        fn (
            GraphPtr,
            mlir.Location.c_type,
            StringRef,
            TuplePtr,
            ArityPtr,
            AttrMapPtr,
        ) -> TuplePtr
    ]("MAXG_graphNewOp")(g, loc.c, name, inputs, out_types, attrs)


# ===----------------------------------------------------------------------===#
# Module support
# ===----------------------------------------------------------------------===#


fn module_to_bytecode(m: mlir.Module, file_name: String) -> Bool:
    return cfunc[fn (mlir.Module.c_type, StringRef) -> Bool](
        "MAXG_moduleToBytecode"
    )(m.c, file_name._strref_dangerous())


# ===----------------------------------------------------------------------===#
# Symbol support
# ===----------------------------------------------------------------------===#


fn symbol_get_graph(s: SymbolPtr) -> GraphPtr:
    return cfunc[fn (SymbolPtr) -> GraphPtr]("MAXG_symbolGetGraph")(s)


fn symbol_to_string(s: SymbolPtr) -> String:
    var len: Int64 = 0
    let ret = cfunc[fn (SymbolPtr, Pointer[Int64]) -> Pointer[Int8]](
        "MAXG_symbolToString"
    )(s, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn tuple_new() -> TuplePtr:
    return cfunc[fn () -> TuplePtr]("MAXG_tupleNew")()


fn tuple_size(t: TuplePtr) -> Int:
    return cfunc[fn (TuplePtr) -> Int]("MAXG_tupleSize")(t)


fn tuple_append_symbol(t: TuplePtr, symbol: SymbolPtr):
    return cfunc[fn (TuplePtr, SymbolPtr) -> NoneType](
        "MAXG_tupleAppendSymbol"
    )(t, symbol)


fn tuple_get_symbol(t: TuplePtr, pos: UInt32) -> SymbolPtr:
    return cfunc[fn (TuplePtr, UInt32) -> SymbolPtr]("MAXG_getSymbol")(t, pos)


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
    m: mlir.Module, dtype: mlir.Type, dims: DynamicVector[Int64], ranked: Bool
) -> mlir.Type:
    return cfunc[
        fn (
            mlir.Module.c_type, mlir.Type.c_type, Bool, Pointer[Int64], Int32
        ) -> mlir.Type.c_type
    ]("MAXG_tensorTypeNew")(
        m.c, dtype.c, ranked, Pointer[Int64](dims.data.value), len(dims)
    )


fn tensor_type_get_dtype(s: SymbolPtr) -> DType:
    let dtype = cfunc[fn (SymbolPtr) -> UInt8]("MAXG_tensorTypeGetDType")(s)
    return DType._from_ui8(dtype.value)


fn tensor_type_get_shape(s: SymbolPtr) -> DynamicVector[Int64]:
    var rank: Int32 = 0
    let dims = cfunc[fn (SymbolPtr, Pointer[Int32]) -> Pointer[Int64]](
        "MAXG_tensorTypeGetShape"
    )(s, Pointer.address_of(rank))
    var dimsVec = DynamicVector[Int64]()
    for i in range(rank):
        dimsVec.append(dims[i])
    return dimsVec


fn tensor_type_is_ranked(s: SymbolPtr) -> Bool:
    return cfunc[fn (SymbolPtr) -> Bool]("MAXG_tensorTypeIsRanked")(s)


fn arity_new() -> ArityPtr:
    return cfunc[fn () -> ArityPtr]("MAXG_arityNew")()


fn arity_size(a: ArityPtr) -> Int:
    return cfunc[fn (ArityPtr) -> Int]("MAXG_aritySize")(a)


fn arity_append_type(a: ArityPtr, t: mlir.Type):
    return cfunc[fn (ArityPtr, mlir.Type.c_type) -> NoneType](
        "MAXG_arityAppendType"
    )(a, t.c)
