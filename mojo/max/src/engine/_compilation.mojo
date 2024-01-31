# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from ._context import *
from ._model_specs import *
from .session import InferenceSession


@value
@register_passable("trivial")
struct FrameworkFormat:
    """Enum-like struct indicating the model framework."""

    alias MAXGraph = FrameworkFormat(0)

    var value: UInt8


@value
@register_passable("trivial")
struct ModelSource:
    """Model source representation that is ABI compatible with the C API's `M_ModelSource`.
    """

    var source: Pointer[NoneType]
    var format: FrameworkFormat


@value
@register_passable("trivial")
struct CCompileConfig:
    """Mojo representation of Engine's CompileConfig pointer.
    This doesn't free the memory on destruction. For memory managed
    option see CompileConfig.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompileConfigFnName = "M_freeCompileConfig"
    alias SetModelSourceFnName = "M_setModelSource"
    alias SetModelPathFnName = "M_setModelPath"
    alias ReplaceOpsFnName = "M_useKernelsFrom"
    alias SetDeviceFnName = "M_setDevice"

    fn set_model_source(
        self, model_source: ModelSource, borrowed lib: DLHandle
    ):
        call_dylib_func(lib, Self.SetModelSourceFnName, self, model_source)

    fn set_model_path(self, path: String, borrowed lib: DLHandle):
        """Sets the path of model to compile."""
        var path_strref = path._strref_dangerous()
        call_dylib_func(lib, Self.SetModelPathFnName, self, path_strref.data)
        path._strref_keepalive()

    fn replace_ops(self, path: String, borrowed lib: DLHandle) raises:
        let status = Status(lib)
        var path_strref = path._strref_dangerous()
        call_dylib_func(
            lib, Self.ReplaceOpsFnName, self, path_strref.data, status.ptr
        )
        path._strref_keepalive()
        if status:
            raise Error(status.__str__())

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompileConfigFnName, self)


struct CompileConfig:
    """Memory managed version of Engine's Compile Config."""

    var ptr: Pointer[CCompileConfig]
    var lib: DLHandle

    alias NewCompileConfigFnName = "M_newCompileConfig"

    fn __init__(inout self, borrowed lib: DLHandle):
        self.ptr = Pointer[CCompileConfig].alloc(1)
        __get_address_as_uninit_lvalue(self.ptr.address) = call_dylib_func[
            CCompileConfig
        ](lib, Self.NewCompileConfigFnName)
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.lib = existing.lib

    fn set_model_source(self, model_source: ModelSource):
        __get_address_as_lvalue(self.ptr.address).set_model_source(
            model_source, self.lib
        )

    fn set_model_path(self, path: String):
        """Sets the path of model to compile."""
        __get_address_as_lvalue(self.ptr.address).set_model_path(path, self.lib)

    fn set_replace_ops_path(self, path: String) raises:
        """Replace Modular kernels with user-defined kernels."""
        __get_address_as_lvalue(self.ptr.address).replace_ops(path, self.lib)

    fn borrow_ptr(self) -> Pointer[CCompileConfig]:
        return self.ptr

    fn __del__(owned self):
        __get_address_as_lvalue(self.ptr.address).free(self.lib)
        _ = __get_address_as_owned_value(self.ptr.address)
        self.ptr.free()


@value
@register_passable("trivial")
struct CCompiledModel:
    """Mojo representation of Engine's AsyncCompiledModel pointer.
    Useful for C inter-op.
    """

    var ptr: DTypePointer[DType.invalid]

    alias FreeCompiledModelFnName = "M_freeCompiledModel"
    alias GetNumInputsFnName = "M_getNumModelInputs"
    alias GetNumOutputsFnName = "M_getNumModelInputs"

    fn num_model_inputs(self, borrowed lib: DLHandle) raises -> Int:
        """Gets the number of inputs of the model."""

        let status = Status(lib)
        let num_inputs = call_dylib_func[Int](
            lib, Self.GetNumInputsFnName, self, status.ptr
        )
        if status:
            raise Error(status.__str__())
        return num_inputs

    fn num_model_outputs(self, borrowed lib: DLHandle) raises -> Int:
        """Gets the number of inputs of the model."""

        let status = Status(lib)
        let num_outputs = call_dylib_func[Int](
            lib, Self.GetNumOutputsFnName, self, status.ptr
        )
        if status:
            raise Error(status.__str__())
        return num_outputs

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompiledModelFnName, self)


@value
struct CompiledModel:
    """Memory managed CompiledModel pointer."""

    var ptr: CCompiledModel
    var lib: DLHandle
    var session: InferenceSession

    alias CompileModelFnName = "M_compileModelSync"
    alias GetModelInputSpecByNameFnName = "M_getModelInputSpecByName"
    alias GetModelOutputSpecByNameFnName = "M_getModelOutputSpecByName"

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = exchange[CCompiledModel](
            existing.ptr, DTypePointer[DType.invalid].get_null()
        )
        self.lib = existing.lib
        self.session = existing.session ^

    fn num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model."""

        return self.ptr.num_model_inputs(self.lib)

    fn get_model_input_names(self) raises -> InputTensorNames:
        """Gets the names of model inputs."""

        return InputTensorNames(self.ptr, self.num_model_inputs(), self.lib)

    fn num_model_outputs(self) raises -> Int:
        """Gets the number of inputs of the model."""

        return self.ptr.num_model_outputs(self.lib)

    fn get_model_output_names(self) raises -> OutputTensorNames:
        """Gets the names of model outputs."""

        return OutputTensorNames(self.ptr, self.num_model_outputs(), self.lib)

    fn borrow_ptr(self) -> CCompiledModel:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
