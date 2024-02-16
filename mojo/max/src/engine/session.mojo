# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from collections.vector import InlinedFixedVector, DynamicVector
from memory.anypointer import AnyPointer
from os.atomic import Atomic
from sys.ffi import DLHandle

from max.graph import Module

from ._context import _Device, RuntimeContext, RuntimeConfig
from ._compilation import (
    CCompiledModel,
    CompileConfig,
    CompiledModel,
    FrameworkFormat,
    ModelSource,
)
from ._engine_impl import _EngineImpl, _get_engine_path
from ._model_impl import CModel
from ._status import Status
from ._tensor_spec_impl import TensorSpec
from ._utils import call_dylib_func


struct _InferenceSessionImpl(Movable):
    var engine: _EngineImpl
    var context: RuntimeContext
    var ref_count: Atomic[DType.int64]

    fn __init__(
        inout self,
        lib_path: String,
        device: _Device,
    ):
        self.engine = _EngineImpl(lib_path)
        let config = RuntimeConfig(self.engine.lib, device)
        self.context = RuntimeContext(config ^, self.engine.lib)
        self.ref_count = 1

    fn __moveinit__(inout self, owned existing: Self):
        self.engine = existing.engine ^
        self.context = existing.context ^
        self.ref_count = existing.ref_count.fetch_add(0)

    fn _compile_model_from_config(
        self,
        owned config: LoadOptions,
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to compile model"

        var compile_config = CompileConfig(self.engine.lib)

        let model_path = config._model_path
        if model_path:
            let path = model_path.value()
            compile_config.set_model_path(path.path._strref_dangerous())
            path.path._strref_keepalive()

        let custom_ops_path = config._custom_ops_path
        if custom_ops_path:
            let path = custom_ops_path.value()
            compile_config.set_replace_ops_path(path.path._strref_dangerous())
            path.path._strref_keepalive()

        let model_source = config._source
        if model_source and model_path:
            raise "give either module source or path"

        if model_source:
            compile_config.set_model_source(model_source.value())

        let spec_count = len(config._input_specs)
        for i in range(spec_count):
            let _spec = config._input_specs[i]
            if _spec.static:
                compile_config.add_input_spec(_spec.static.value())
            else:
                let dtype = _spec.dtype
                compile_config.add_input_spec(_spec.dynamic, dtype)

        compile_config.set_torch_input_specs()

        let status = Status(self.engine.lib)
        let compile_ptr = compile_config.borrow_ptr()
        let compiled_model_ptr = call_dylib_func[CCompiledModel](
            self.engine.lib,
            CompiledModel.CompileModelFnName,
            context,
            compile_ptr,
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        let model = CompiledModel(
            compiled_model_ptr, self.engine.lib, session ^
        )
        _ = compile_config ^

        # We could borrow config and don't do this,
        # but internally C APi will take ownership of compile_config ptr
        # and mutates it to null. This will convey the intention that we are
        # mutating something we own. There is no need for caller of this
        # to have the mutated value now. That negates the need for inout.
        _ = config ^

        return model ^

    fn _init_model(
        self,
        owned compiled_model: CompiledModel,
        owned session: InferenceSession,
    ) raises -> Model:
        let status = Status(self.engine.lib)
        let model_ptr = call_dylib_func[CModel](
            self.engine.lib,
            Model._InitModelFnName,
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        model_ptr.await_model(self.engine.lib)

        let model = Model(
            self.context.borrow_ptr(),
            model_ptr,
            self.engine.lib,
            session ^,
            compiled_model ^,
        )
        return model ^

    fn load_model(
        self,
        owned config: LoadOptions,
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes the model.
        """
        let compiled_model = self._compile_model_from_config(
            config ^, session.copy()
        )

        return self._init_model(compiled_model ^, session ^)

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        spec: TensorSpec,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, spec, self.engine.lib, session ^)

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        shape: Optional[DynamicVector[Optional[Int64]]],
        dtype: DType,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, shape, dtype, self.engine.lib, session ^)

    fn new_tensor_map(
        self, owned session: InferenceSession
    ) raises -> TensorMap:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor map"
        return TensorMap(self.context.borrow_ptr(), self.engine.lib, session ^)

    fn new_borrowed_tensor_value[
        type: DType
    ](
        self, owned session: InferenceSession, tensor: Tensor[type]
    ) raises -> Value:
        """Create a new Value representing data borrowed from given tensor."""
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor value"
        return Value._new_borrowed_tensor[type](
            self.context.borrow_ptr(), self.engine.lib, session ^, tensor
        )

    fn new_bool_value(
        self, owned session: InferenceSession, value: Bool
    ) raises -> Value:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create bool value"
        return Value._new_bool(
            self.context.borrow_ptr(), self.engine.lib, session ^, value
        )

    fn new_list_value(self, owned session: InferenceSession) raises -> Value:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create list value"
        return Value._new_list(
            self.context.borrow_ptr(), self.engine.lib, session ^
        )


@value
struct _Specs(CollectionElement):
    var static: Optional[TensorSpec]

    alias dynamic_type = Optional[DynamicVector[Optional[Int64]]]
    var dynamic: Self.dynamic_type
    var dtype: DType

    fn __init__(inout self, spec: TensorSpec):
        self.static = spec
        self.dynamic = None
        self.dtype = spec.dtype()

    fn __init__(
        inout self, spec: Optional[DynamicVector[Optional[Int64]]], dtype: DType
    ):
        self.static = None
        self.dynamic = spec
        self.dtype = dtype

    fn __init__(inout self, spec: NoneType, dtype: DType):
        self.static = None
        self.dynamic = None
        self.dtype = dtype


@value
struct LoadOptions(CollectionElement):
    var _source: Optional[ModelSource]
    var _model_path: Optional[Path]
    var _custom_ops_path: Optional[Path]
    var _input_specs: DynamicVector[_Specs]

    fn __init__(inout self):
        self._source = None
        self._model_path = None
        self._custom_ops_path = None
        self._input_specs = DynamicVector[_Specs]()

    fn _set_model_source(inout self, module: Module):
        """Specifies the Max Graph Module to load model from.
           Use either this function or `set_model_path` function
           to specify model source.

        Args:
            module: Max Graph module.
        """
        let mlir_module = module._module
        self._source = ModelSource(mlir_module.c.ptr, FrameworkFormat.MAXGraph)

    fn _set_model_path(inout self, path: Path):
        """Specifies the loaction in filesystem to load model from.
           Use either this function or `set_model_source` function
           to specify model source.

        Args:
            path: Path of the model on disk.

        """
        self._model_path = path

    fn set_custom_ops_path(inout self, path: Path) raises:
        """Replace Modular kernels in given model with user-defined kernels.

        Args:
            path: Path to mojo custom op package.
        """
        self._custom_ops_path = path

    fn add_input_spec(inout self, spec: TensorSpec):
        """Add valid input specs for model to be given at compile time.
           Only applicable for PyTorch.

        Args:
            spec: Spec for the input.
        """
        self._input_specs.push_back(_Specs(spec))

    fn add_input_spec(
        inout self,
        shape: _Specs.dynamic_type,
        dtype: DType,
    ):
        """Add valid input specs for model to be given at compile time.
           Only applicable for PyTorch.

        Args:
            shape: Shape of the input.
            dtype: Datatype of the input.
        """
        self._input_specs.push_back(_Specs(shape, dtype))

    fn add_input_specs(
        inout self,
        specs: DynamicVector[TensorSpec],
    ) raises:
        """Add valid input specs for model to be given at compile time.
           Only applicable for PyTorch.

        Args:
            specs: Specs for the input.
        """

        for i in range(len(specs)):
            self._input_specs.push_back(_Specs(specs[i]))

    fn add_input_specs(
        inout self,
        shapes: DynamicVector[_Specs.dynamic_type],
        dtypes: InlinedFixedVector[DType],
    ) raises:
        """Add valid input specs for model to be given at compile time.
           Only applicable for PyTorch.

        Args:
            shapes: Shapes of the input.
            dtypes: Datatypes of the input.
        """
        for i in range(len(shapes)):
            self._input_specs.push_back(_Specs(shapes[i], dtypes[i]))


@value
struct SessionOptions:
    var _device: _Device

    fn __init__(inout self):
        self = Self(_Device.CPU)

    fn _set_device(inout self, device: _Device):
        self._device = device


@value
@register_passable
struct InferenceSession:
    var ptr: AnyPointer[_InferenceSessionImpl]

    fn __init__(options: SessionOptions = SessionOptions()) raises -> Self:
        let path = _get_engine_path()
        let self = Self._allocateAndInit(path, options._device)
        return Self {ptr: self}

    @staticmethod
    fn _allocateAndInit(
        lib_path: String,
        device: _Device,
    ) raises -> AnyPointer[_InferenceSessionImpl]:
        let ptr = AnyPointer[_InferenceSessionImpl].alloc(1)
        __get_address_as_uninit_lvalue(ptr.value).__init__(lib_path, device)
        return ptr

    fn copy(self) -> Self:
        _ = __get_address_as_lvalue(self.ptr.value).ref_count.fetch_add(1)
        return Self {ptr: self.ptr}

    fn load_model(
        self, path: Path, config: Optional[LoadOptions] = None
    ) raises -> Model:
        """Compile and Initialize AI model with Max
           engine with given path and config.

        Args:
            path: Location of model in filesystem.
            config: Configurations need for compiling model.

        Returns:
            Initialized model ready for inference.

        """
        var load_config: LoadOptions
        if config:
            load_config = config.value()
        else:
            load_config = LoadOptions()
        load_config._set_model_path(path)
        return __get_address_as_lvalue(self.ptr.value).load_model(
            load_config ^, self.copy()
        )

    fn load_model(
        self, module: Module, config: Optional[LoadOptions] = None
    ) raises -> Model:
        """Compile and Initialize AI model with Max
           engine with given Max Graph module and config.

        Args:
            module: Max Graph module.
            config: Configurations need for compiling model.

        Returns:
            Initialized model ready for inference.

        """
        var load_config: LoadOptions
        if config:
            load_config = config.value()
        else:
            load_config = LoadOptions()
        load_config._set_model_source(module)
        return __get_address_as_lvalue(self.ptr.value).load_model(
            load_config ^, self.copy()
        )

    fn get_as_engine_tensor_spec(
        self, name: String, spec: TensorSpec
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with Max Engine.

        Args:
            name: Name of the Tensor.
            spec: Tensor specification in Mojo TensorSpec format.

        Returns:
           EngineTensorSpec to be used with Max Engine APIs.

        """
        return __get_address_as_lvalue(
            self.ptr.value
        ).get_as_engine_tensor_spec(name, spec, self.copy())

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        shape: Optional[DynamicVector[Optional[Int64]]],
        dtype: DType,
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with Max Engine.

        Args:
            name: Name of the Tensor.
            shape: Shape of the Tensor.
                   Dynamic Dimensions can be represented with None and for
                   Dynamic Rank Tensor use None as value for shape.
            dtype: DataType of the Tensor.

        Returns:
            EngineTensorSpec to be used with Max Engine APIs.
        """
        return __get_address_as_lvalue(
            self.ptr.value
        ).get_as_engine_tensor_spec(name, shape, dtype, self.copy())

    fn new_tensor_map(self) raises -> TensorMap:
        return __get_address_as_lvalue(self.ptr.value).new_tensor_map(
            self.copy()
        )

    fn new_borrowed_tensor_value[
        type: DType
    ](self, tensor: Tensor[type]) raises -> Value:
        """Create a new Value representing data borrowed from given tensor."""
        return __get_address_as_lvalue(
            self.ptr.value
        ).new_borrowed_tensor_value(self.copy(), tensor)

    fn new_bool_value(self, value: Bool) raises -> Value:
        """Create a new Value representing a Bool."""
        return __get_address_as_lvalue(self.ptr.value).new_bool_value(
            self.copy(), value
        )

    fn new_list_value(self) raises -> Value:
        """Create a new Value representing an empty list."""
        return __get_address_as_lvalue(self.ptr.value).new_list_value(
            self.copy()
        )

    fn __del__(owned self):
        if __get_address_as_lvalue(self.ptr.value).ref_count.fetch_sub(1) != 1:
            # There are others holding reference to this session. Keep the
            # session alive and let other reference holders deal with
            # managing it.
            return

        # Now we know that there is only active reference to this session
        # and that is held by us. So go ahead and call destructor on it.
        _ = __get_address_as_owned_value(self.ptr.value)
        self.ptr.free()
