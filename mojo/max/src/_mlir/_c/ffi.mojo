# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort, getenv
from pathlib import Path
from sys.ffi import RTLD, DLHandle, _get_dylib_function, external_call
from sys.param_env import env_get_string, is_defined

from builtin.builtin_list import _LITRefPackHelper
from memory import UnsafePointer

from utils import StringRef


# Init fns inspired by gpu.host._utils
fn _init_dylib(ignored: UnsafePointer[NoneType]) -> UnsafePointer[NoneType]:
    alias mlirc_dylib = env_get_string["MLIRC_DYLIB", ".graph_lib"]()
    var mof_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8]
    ](StringRef(mlirc_dylib))

    if not mof_lib_path_str_ptr:
        abort("cannot get graph library location from modular.cfg")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getMAXConfigValue` so that it can be destroyed by Mojo.
    var mof_lib_path = String._from_bytes(mof_lib_path_str_ptr)

    if not Path(mof_lib_path).exists():
        abort("cannot load graph library from " + mof_lib_path)

    var ptr = UnsafePointer[DLHandle].alloc(1)
    ptr.init_pointee_move(DLHandle(mof_lib_path, RTLD.NOW | RTLD.GLOBAL))
    return ptr.bitcast[NoneType]()


fn _destroy_dylib(ptr: UnsafePointer[NoneType]):
    ptr.bitcast[DLHandle]()[].close()
    ptr.free()


@always_inline
fn MLIR_func[
    name: StringLiteral, T: AnyTrivialRegType, *Args: AnyType
](*args: *Args) -> T:
    var loaded_args_pack = _LITRefPackHelper(args._value).get_loaded_kgen_pack()

    @parameter
    if not is_defined["MLIRCAPI_LINKED"]():
        var f = _get_dylib_function[
            "MOF_LIB",
            name,
            _init_dylib,
            _destroy_dylib,
            fn (__type_of(loaded_args_pack)) -> T,
        ]()
        var ptr = UnsafePointer.address_of(f).bitcast[
            UnsafePointer[NoneType]
        ]()[]
        if not ptr:
            abort("cannot load " + String(name) + " from graph library")
        return f(loaded_args_pack)
    else:
        return external_call[name, T](loaded_args_pack)
