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
from sys import external_call
from ._tensor_spec_impl import CTensorSpec


@value
@register_passable("trivial")
struct FrameworkFormat:
    """Enum-like struct indicating the model framework."""

    alias MAXGraph = FrameworkFormat(0)

    var value: UInt8


@value
@register_passable("trivial")
struct ModelSource(CollectionElement):
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
    alias SetTorchInputSpecsFnName = "M_setTorchInputSpecs"

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

    fn set_torch_input_specs(
        self,
        torch_lib: DLHandle,
        specs_ptr: Pointer[CTensorSpec],
        spec_count: Int,
    ):
        call_dylib_func(
            torch_lib,
            Self.SetTorchInputSpecsFnName,
            self,
            specs_ptr,
            spec_count,
        )

    fn free(self, borrowed lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompileConfigFnName, self)


struct CompileConfig:
    """Memory managed version of Engine's Compile Config."""

    var ptr: Pointer[CCompileConfig]
    var lib: DLHandle
    var torch_lib: Optional[DLHandle]

    var input_specs: AnyPointer[EngineTensorSpec]
    var input_specs_c_ptr: Pointer[CTensorSpec]
    var input_spec_count: Int
    var input_spec_capacity: Int
    alias initial_capacity = 5

    alias NewCompileConfigFnName = "M_newCompileConfig"

    fn __init__(inout self, borrowed lib: DLHandle):
        self.ptr = Pointer[CCompileConfig].alloc(1)
        __get_address_as_uninit_lvalue(self.ptr.address) = call_dylib_func[
            CCompileConfig
        ](lib, Self.NewCompileConfigFnName)
        self.lib = lib

        # Pick a reasonable intitial value
        self.input_spec_capacity = Self.initial_capacity
        self.input_specs = AnyPointer[EngineTensorSpec].alloc(
            self.input_spec_capacity
        )
        self.input_specs_c_ptr = Pointer[CTensorSpec].alloc(
            self.input_spec_capacity
        )
        self.input_spec_count = 0
        self.torch_lib = Self._get_torch_lib()

    @staticmethod
    fn _get_torch_lib() -> Optional[DLHandle]:
        # Since we only need to open this library for this case we
        # can lazy load it here.
        let torch_ext_lib_path_str_ptr = external_call[
            "KGEN_CompilerRT_getConfigValue", DTypePointer[DType.int8]
        ]("max.torch_ext_lib")

        # This transfers ownership of the underlying data buffer allocated in
        # `KGEN_CompilerRT_getConfigValue` so that it can be destroyed by Mojo.
        let pathlen = len(StringRef(torch_ext_lib_path_str_ptr))
        let torch_ext_lib_path = String(
            torch_ext_lib_path_str_ptr, pathlen + 1
        )  # account for the terminator

        if not torch_ext_lib_path:
            return None

        if not Path(torch_ext_lib_path).exists():
            return None

        return DLHandle(torch_ext_lib_path)

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.lib = existing.lib
        self.input_specs = existing.input_specs
        self.input_specs_c_ptr = existing.input_specs_c_ptr
        self.input_spec_count = existing.input_spec_count
        self.input_spec_capacity = existing.input_spec_capacity
        self.torch_lib = existing.torch_lib

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

    fn set_torch_input_specs(self) raises:
        if self.input_spec_count == 0:
            return

        if not self.torch_lib:
            raise "cannot find torch extension libraries"

        __get_address_as_lvalue(self.ptr.address).set_torch_input_specs(
            self.torch_lib.value(),
            self.input_specs_c_ptr,
            self.input_spec_count,
        )

    fn add_input_spec(inout self, owned spec: EngineTensorSpec):
        # We don't have a collection type for Movable only objects
        # So we keep track of both EngineTensorSpec to clear at the end
        # as well as an array of CTensorSpec to pass to C API.

        # If we have reached capacity realloc and move existing values there.
        if self.input_spec_count == self.input_spec_capacity:
            # Double the capacity and allocate new ptrs.
            self.input_spec_capacity *= 2
            let input_specs = AnyPointer[EngineTensorSpec].alloc(
                self.input_spec_capacity
            )
            let input_specs_c_ptr = Pointer[CTensorSpec].alloc(
                self.input_spec_capacity
            )

            # Move old values to new location and delete the
            # old memory.
            for i in range(self.input_spec_count):
                _ = __get_address_as_owned_value(
                    (input_specs_c_ptr + i).address
                )
                (input_specs + i).emplace_value(
                    (self.input_specs + i).take_value()
                )
                let inner_ptr = __get_address_as_lvalue(
                    (input_specs + i).value
                )._borrow_ptr()
                input_specs_c_ptr.store(i, inner_ptr)

            self.input_specs_c_ptr.free()
            self.input_specs.free()
            self.input_specs = input_specs
            self.input_specs_c_ptr = input_specs_c_ptr

        # Keep both engine tensor spec and its inner C spec.
        (self.input_specs + self.input_spec_count).emplace_value(spec ^)
        let inner_ptr = __get_address_as_lvalue(
            (self.input_specs + self.input_spec_count).value
        )._borrow_ptr()
        self.input_specs_c_ptr.store(self.input_spec_count, inner_ptr)
        self.input_spec_count += 1

    fn borrow_ptr(self) -> Pointer[CCompileConfig]:
        return self.ptr

    fn take_ptr(inout self) -> Pointer[CCompileConfig]:
        let ptr = self.ptr
        self.ptr = Pointer[CCompileConfig]()
        return ptr

    fn _del_specs(self):
        for i in range(self.input_spec_count):
            _ = (self.input_specs + i).take_value()

        # We don't need to call destructor for inner c ptr
        self.input_specs_c_ptr.free()
        self.input_specs.free()

    fn __del__(owned self):
        self._del_specs()

        if self.torch_lib:
            var torch = self.torch_lib.value()
            torch._del_old()

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
    alias GetNumOutputsFnName = "M_getNumModelOutputs"

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
        """Gets the number of outputs of the model."""

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

    fn get_model_input_names(self) raises -> DynamicVector[String]:
        """Gets the names of model inputs."""

        let names = InputTensorNames(
            self.ptr, self.num_model_inputs(), self.lib
        )
        var name_vec = DynamicVector[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.push_back(names[i])
        return name_vec

    fn num_model_outputs(self) raises -> Int:
        """Gets the number of outputs of the model."""

        return self.ptr.num_model_outputs(self.lib)

    fn get_model_output_names(self) raises -> DynamicVector[String]:
        """Gets the names of model outputs."""

        let names = OutputTensorNames(
            self.ptr, self.num_model_outputs(), self.lib
        )
        var name_vec = DynamicVector[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.push_back(names[i])
        return name_vec

    fn borrow_ptr(self) -> CCompiledModel:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
