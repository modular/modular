# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import Pointer
from os import getenv
from tensor import Tensor
from sys.ffi import RTLD, DLHandle, _get_dylib_function


struct _Attr:
    """Opaque data type mapping to M_Attr in the Builder C API."""

    pass


struct _AttrMap:
    """Opaque data type mapping to M_AttrMap in the Builder C API."""

    pass


struct _Graph:
    """Opaque data type mapping to M_Graph in the Builder C API."""

    pass


struct _Loc:
    """Opaque data type mapping to M_Loc in the Builder C API."""

    pass


struct _Module:
    """Opaque data type mapping to M_Module in the Builder C API."""

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


struct _Type:
    """Opaque data type mapping to M_Type in the Builder C API."""

    pass


struct _TensorTuple:
    """Opaque data type mapping to M_TensorTuple in the MOF C API."""

    pass


alias AttrPtr = Pointer[_Attr]
alias AttrMapPtr = Pointer[_AttrMap]
alias GraphPtr = Pointer[_Graph]
alias LocPtr = Pointer[_Loc]
alias ModulePtr = Pointer[_Module]
alias TuplePtr = Pointer[_Tuple]
alias SymbolPtr = Pointer[_Symbol]
alias ArityPtr = Pointer[_Arity]
alias TypePtr = Pointer[_Type]
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
        print("cannot get the location of graph library from modular.cfg")
        trap()

    if not Path(mof_lib_path).exists():
        print("cannot load graph library. Checked path " + mof_lib_path)
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
fn MOBI_func[T: AnyRegType](name: StringRef) -> T:
    var f = _get_dylib_function["MOF_LIB", _init_dylib, _destroy_dylib, T](name)
    let ptr = Pointer.address_of(f).bitcast[Pointer[NoneType]]().load()
    if not ptr:
        print("cannot load " + String(name) + " from shared library")
        trap()
    return f


# Note: Keep sections below in sync with mobicc.mojo, including order, grouping
# and naming.

# Note: Please keep the following naming convention: conept_function. For
# example graph_new, etc.


# ===----------------------------------------------------------------------===#
# Attribute factories
# ===----------------------------------------------------------------------===#


fn attr_new_tensor[
    T: CollectionElement
](
    module: ModulePtr,
    name: StringRef,
    data: DynamicVector[T],
    type: TypePtr,
    is_owned: Bool,
) -> AttrPtr:
    return MOBI_func[
        fn (ModulePtr, StringRef, AnyPointer[T], TypePtr, Bool) -> AttrPtr
    ]("MOBI_attrNewTensor")(module, name, data.data, type, is_owned)


fn attr_new_tensor(
    module: ModulePtr,
    name: StringRef,
    data: DTypePointer[DType.invalid],
    type: TypePtr,
    is_owned: Bool,
) -> AttrPtr:
    return MOBI_func[
        fn (
            ModulePtr, StringRef, DTypePointer[DType.invalid], TypePtr, Bool
        ) -> AttrPtr
    ]("MOBI_attrNewTensor")(module, name, data, type, is_owned)


fn attr_new_tensor_from_file(
    module: ModulePtr,
    name: StringRef,
    file_name: StringRef,
    type: TypePtr,
) -> AttrPtr:
    return MOBI_func[fn (ModulePtr, StringRef, StringRef, TypePtr) -> AttrPtr](
        "MOBI_attrNewTensorFromFile"
    )(module, name, file_name, type)


fn attr_new_string(
    module: ModulePtr,
    name: StringRef,
    value: StringRef,
) -> AttrPtr:
    return MOBI_func[fn (ModulePtr, StringRef, StringRef) -> AttrPtr](
        "MOBI_attrNewString"
    )(module, name, value)


# ===----------------------------------------------------------------------===#
# AttrMap support
# ===----------------------------------------------------------------------===#


fn attr_map_new() -> AttrMapPtr:
    return MOBI_func[fn () -> AttrMapPtr]("MOBI_attrMapNew")()


fn attr_map_add_attr(inout m: AttrMapPtr, a: AttrPtr):
    return MOBI_func[fn (AttrMapPtr, AttrPtr) -> NoneType](
        "MOBI_attrMapAddAttr"
    )(m, a)


fn attr_map_size(m: AttrMapPtr) -> Int:
    return MOBI_func[fn (AttrMapPtr) -> Int]("MOBI_attrMapSize")(m)


# ===----------------------------------------------------------------------===#
# Graph support
# ===----------------------------------------------------------------------===#


fn graph_new(
    inout module: ModulePtr,
    loc: LocPtr,
    name: StringRef,
    inTypes: ArityPtr,
    outTypes: ArityPtr,
) -> GraphPtr:
    return MOBI_func[
        fn (ModulePtr, LocPtr, StringRef, ArityPtr, ArityPtr) -> GraphPtr
    ]("MOBI_graphNew")(module, loc, name, inTypes, outTypes)


fn graph_get_module(graph: GraphPtr) -> ModulePtr:
    return MOBI_func[fn (GraphPtr) -> ModulePtr]("MOBI_graphGetModule")(graph)


fn graph_get_arg(
    graph: GraphPtr,
    pos: UInt32,
) -> SymbolPtr:
    return MOBI_func[fn (GraphPtr, UInt32) -> SymbolPtr]("MOBI_graphGetArg")(
        graph, pos
    )


fn graph_new_op(
    graph: GraphPtr,
    loc: LocPtr,
    name: StringRef,
    inputs: TuplePtr,
    outTypes: ArityPtr,
    attrs: AttrMapPtr,
) -> TuplePtr:
    return MOBI_func[
        fn (
            GraphPtr, LocPtr, StringRef, TuplePtr, ArityPtr, AttrMapPtr
        ) -> TuplePtr
    ]("MOBI_graphNewOp")(graph, loc, name, inputs, outTypes, attrs)


# ===----------------------------------------------------------------------===#
# Location helpers
# ===----------------------------------------------------------------------===#


fn loc_new_unknown(module: ModulePtr) -> LocPtr:
    return MOBI_func[fn (ModulePtr) -> LocPtr]("MOBI_locNewUnknown")(module)


# ===----------------------------------------------------------------------===#
# Module support
# ===----------------------------------------------------------------------===#


fn module_new() -> ModulePtr:
    return MOBI_func[fn () -> ModulePtr]("MOBI_moduleNew")()


fn module_verify(module: ModulePtr) -> Bool:
    return MOBI_func[fn (ModulePtr) -> Bool]("MOBI_moduleVerify")(module)


fn module_to_string(module: ModulePtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (ModulePtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_moduleToString"
    )(module, Pointer[Int64].address_of(len))
    debug_assert(
        ret[len.to_int()] == 0, "String expects null-terminated buffers"
    )
    return String(ret, len.to_int())


fn module_to_bytecode(module: ModulePtr, file_name: String) -> Bool:
    return MOBI_func[fn (ModulePtr, StringRef) -> Bool](
        "MOBI_moduleToBytecode"
    )(module, file_name._strref_dangerous())


# ===----------------------------------------------------------------------===#
# Symbol support
# ===----------------------------------------------------------------------===#


fn symbol_get_graph(symbol: SymbolPtr) -> GraphPtr:
    return MOBI_func[fn (SymbolPtr) -> GraphPtr]("MOBI_symbolGetGraph")(symbol)


fn symbol_to_string(symbol: SymbolPtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (SymbolPtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_symbolToString"
    )(symbol, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn tuple_new() -> TuplePtr:
    return MOBI_func[fn () -> TuplePtr]("MOBI_tupleNew")()


fn tuple_size(tup: TuplePtr) -> Int:
    return MOBI_func[fn (TuplePtr) -> Int]("MOBI_tupleSize")(tup)


fn tuple_add_symbol(inout tup: TuplePtr, symbol: SymbolPtr):
    return MOBI_func[fn (TuplePtr, SymbolPtr) -> NoneType](
        "MOBI_tupleAddSymbol"
    )(tup, symbol)


fn tuple_get_symbol(tup: TuplePtr, pos: UInt32) -> SymbolPtr:
    return MOBI_func[fn (TuplePtr, UInt32) -> SymbolPtr]("MOBI_getSymbol")(
        tup, pos
    )


# ===----------------------------------------------------------------------===#
# Type helpers
# ===----------------------------------------------------------------------===#


fn dtype_new(m: ModulePtr, dtype: DType) -> TypePtr:
    return MOBI_func[fn (ModulePtr, UInt8) -> TypePtr]("MOBI_dTypeNew")(
        m, dtype._as_i8()
    )


fn dim_type_new_dynamic() -> Int64:
    return MOBI_func[fn () -> Int64]("MOBI_dimTypeNewDynamic")()


fn tensor_type_new(
    m: ModulePtr, dtype: TypePtr, dims: DynamicVector[Int64], ranked: Bool
) -> TypePtr:
    return MOBI_func[
        fn (ModulePtr, TypePtr, Bool, Pointer[Int64], Int32) -> TypePtr
    ]("MOBI_tensorTypeNew")(
        m, dtype, ranked, Pointer[Int64](dims.data.value), len(dims)
    )


fn tensor_type_get_dtype(s: SymbolPtr) -> DType:
    let dtype = MOBI_func[fn (SymbolPtr) -> UInt8]("MOBI_tensorTypeGetDType")(s)
    return DType._from_ui8(dtype.value)


fn tensor_type_get_shape(s: SymbolPtr) -> DynamicVector[Int64]:
    var rank: Int32 = 0
    let dims = MOBI_func[fn (SymbolPtr, Pointer[Int32]) -> Pointer[Int64]](
        "MOBI_tensorTypeGetShape"
    )(s, Pointer.address_of(rank))
    var dimsVec = DynamicVector[Int64]()
    for i in range(rank):
        dimsVec.append(dims[i])
    return dimsVec


fn tensor_type_is_ranked(s: SymbolPtr) -> Bool:
    return MOBI_func[fn (SymbolPtr) -> Bool]("MOBI_tensorTypeIsRanked")(s)


fn type_to_string(type: TypePtr) -> String:
    var len: Int64 = 0
    let ret = MOBI_func[fn (TypePtr, Pointer[Int64]) -> Pointer[Int8]](
        "MOBI_typeToString"
    )(type, Pointer[Int64].address_of(len))
    return String(ret, len.to_int())


fn arity_new() -> ArityPtr:
    return MOBI_func[fn () -> ArityPtr]("MOBI_arityNew")()


fn arity_size(tup: ArityPtr) -> Int:
    return MOBI_func[fn (ArityPtr) -> Int]("MOBI_aritySize")(tup)


fn arity_add_type(inout arity: ArityPtr, t: TypePtr):
    return MOBI_func[fn (ArityPtr, TypePtr) -> NoneType]("MOBI_arityAddType")(
        arity, t
    )
