//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines types in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPES_TD
#define GENERICML_MGPDIALECT_MGPTYPES_TD

include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.td"
include "MLRT/GraphRT/MEFDialect/MEFTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Support/MDialect/MAttrs.td"

class MGP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MGPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def MGP_ContextType : MGP_Type<"Context", "context",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of execution contexts holding a number of state slots.";
  let description = [{
    Contexts may be created by the 'init' entry point of a model, then reused
    by each call to the model's 'execute' entry point.

    Contexts may contain any number of 'state slots'. A state slot holds
    arbitrary data needed by operations or entire framework runtimes, such as
    attributes, scratch space, pre-computed data or cached buffers. Each op may
    declare its own state slot structure, and each call to that op in the graph
    (an 'operation') may have its own instance of that structure.

    An operator such as 'lib.foo' is generally two primitives: 'libp.foo.setup',
    invoked in 'init' to allocate and initialize a state slot, and 'libp.foo',
    which accesses that slot during 'execute'.

    Contexts are represented by `M::MGP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mgp.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def MGP_DTypeType : MGP_Type<"DType", "dtype"> {
  let summary = "Runtime representation of the element type of a CPU tensor.";
  let description = [{
    A compact runtime encoding of an element type, implemented by M::DType.
    Only intended for int and float types common across modern CPUs.

    Examples:
    ```mlir
    !mgp.dtype   // A uint8_t encoding of an element type at runtime.
    ```
  }];

  let extraClassDeclaration = [{
    /// Returns the DType representing the given MLIR IntegerType, FloatType
    /// or NoneType. The NoneType is represented as unknown. Returns an error
    /// if the MLIR type has no representation as a DType.
    static ErrorOr<std::optional<DType>> fromMlirType(Type mlirType);

    /// Returns the MLIR IntegerType, FloatType or NoneType representing the
    /// given optDType. The unknown value is represented as the NoneType MLIR
    /// type. Returns an error if the DType has no representation as an MLIR
    /// type.
    static ErrorOr<Type> toMLIRType(MLIRContext *context, std::optional<DType> optDType);

    /// Returns error if the given optDType cannot be represented as an
    /// MLIR IntegerType, FloatType or NoneType. Otherwise returns success.
    static ErrorOrSuccess hasMLIRType(std::optional<DType> optDType);

    /// Returns the 'raw form' of optDType. This will be 0 if DType is unknown,
    /// otherwise the DType's value.
    static uint8_t asRawDType(std::optional<DType> optDType);

    /// Return the 'optional form' of rawDType. This will be none if the
    /// rawDType is 0, otherwise the DType with rawDType as value.
    static std::optional<DType> fromRawDType(uint8_t rawDType);
  }];
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

def MGP_List : MGP_Type<"List", "list",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of immutable list container.";
  let description = [{
    This type represents an immutable list of elements.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Opaque
//===----------------------------------------------------------------------===//

def MGP_OpaqueType : MGP_Type<"Opaque", "opaque",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque MGP type";
  let description = [{
    Can represent anything; backed by an AnyAsyncValue.

    Example:
    ```mlir
    !mgp.opaque
    !mgp.opaque<{foo = "bar"}>
    ```
  }];

  let parameters = (ins "StringAttr": $symbol,
                        OptionalParameter<"std::optional<DictionaryAttr>">:$metadata);
  let assemblyFormat = "`<` $symbol (`,` $metadata^)? `>`";
}

//===----------------------------------------------------------------------===//
// DeviceContextType
//===----------------------------------------------------------------------===//

def MGP_DeviceContextType : MGP_Type<"DeviceContext", "device_context",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "MGP representation of the driver DeviceContext";
  let description = [{
    Essentially opaque to MGP, but represented directly so it is simpler to
    introspect.

    Examples:
    ```mlir
    // A device context for CUDA gpu 3.
    !mgp.device_context<<"gpu", 3>>
    ```
  }];

  let parameters = (ins M_DeviceRefAttr:$device);
  let assemblyFormat = "`<` $device `>`";
}

//===----------------------------------------------------------------------===//
// DeviceGraphType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphType : MGP_Type<"DeviceGraph", "device_graph",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of a constructed device graph.";
  let description = [{
    Represents a constructed device-side graph (e.g. a CUDA graph) that can be
    replayed via `mgp.device_graph.execute`.

    The `FunctionType` signature's inputs are buffers the caller copies into
    the graph's frozen-input addresses at replay; results are buffers the graph
    writes at replay. Both are operands of `mgp.device_graph.execute`; the
    type-level split lets the synthesis pass and verifiers tell read-only from
    written operands without inspecting the build body. The target device is
    the device of the signature's `!mgp.buffer` types.

    Example:
    ```mlir
    !mgp.device_graph<
        (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) ->
        (!mgp.buffer<<"gpu", 0>>)>
    ```
  }];

  let parameters = (ins "FunctionType":$signature);
  let assemblyFormat = "`<` $signature `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::FunctionType":$signature)>,
    TypeBuilder<(ins "mlir::TypeRange":$inputs, "mlir::TypeRange":$results)>,
    TypeBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$results)>,
  ];
}

//===----------------------------------------------------------------------===//
// DeviceGraphBuilderType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphBuilderType : MGP_Type<"DeviceGraphBuilder",
  "device_graph.builder",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Builder that records nodes into a device graph under construction.";
  let description = [{
    An opaque handle to a device-graph builder. Created in the build body of
    `mgp.device_graph.create` (from the body's `!mgp.device_context`) and
    threaded as the launch argument of `mogg.call.build_graph`, in place of the
    `!mogg.context` that `mogg.call.execute` would take. Each kernel's
    `build_graph` Mojo method records a node against the builder rather than
    dispatching immediately; once every node is recorded the builder is
    instantiated into an `!mgp.device_graph`.

    Example:
    ```mlir
    !mgp.device_graph.builder
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphNodeType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphNodeType : MGP_Type<"DeviceGraphNode", "device_graph.node",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque handle to a node recorded into a device graph.";
  let description = [{
    An opaque handle to a single node recorded into a device graph via a
    kernel's `build_graph` method. Produced by `mogg.call.build_graph` and
    propagated out of the recording stub. The handle carries no structure yet;
    it exists so that future cross-stub dependency tracking can name prior
    nodes (a `depends_on` operand list) without inspecting the build body.

    Example:
    ```mlir
    !mgp.device_graph.node
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCacheType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphCacheType : MGP_Type<"DeviceGraphCache",
  "device_graph.cache",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque handle to a model's device-graph cache.";
  let description = [{
    An opaque handle to the cache of instantiated device graphs belonging to a
    model. Produced by `mgp.device_graph.cache` from the model's `!mgp.context`
    and consumed by `mgp.device_graph.create`, which reuses a cached graph
    instead of rebuilding one when the key derived from its inputs hits.

    There is one cache per model: it lives in the state context, so it is
    created on first use and outlives any single execution.

    Example:
    ```mlir
    !mgp.device_graph.cache
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TensorSpecType
//===----------------------------------------------------------------------===//

def MGP_TensorSpecType : MGP_Type<"TensorSpec", "tensor_spec", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Type of tensor specs.";
  let description = [{
    `!mgp.tensor_spec` represents the shape and dtype of a tensor.

    The rank is always static, implied by the number of dimensions (ie the
    number of `x` delimiters). Only ranks up to kMaxRank are supported
    (currently 8). A rank of zero denotes 'scalars', though in practice this
    has the same runtime representation as rank 1, dimension 1.

    Dynamic (aka unknown) dimensions and dtypes are supported:
     - Each dimension may be static (an unsigned integer literal) or dynamic
       (denoted by `?`). Symbolic dimensions are not expressible in MGP.
     - The dtype may be static (using the DType::getFromString syntax) or
       dynamic (denoted by`unk`).

    DTypes can represent most variations of MLIR IntegerType and FloatTypes.
    However:
     - Signless and non power-of-two MLIR IntegerTypes have no DType
       representation.
     - The complex and a few float DTypes have no MLIR Type representation.
     - The 'invalid' (zero) DType is never permitted.
    If a tensor spec is given a DType we also require it to be representable as
    an MLIR type.

    This type implements ShapedTypeInterface. We'll use 'dims'/'dtype' when
    referring to our internal representation, and 'shape'/'element type' when
    referring to the ShapedTypeInterface representation.

    Representations in flight:
     - For dimensions:
        - 'optional' form: std::optional<uint64_t>, where nullopt denotes
          a dynamic dimension.
        - 'signed' form: int64_t, where ShapedType::kDynamicSize denotes
          a dynamic dimension. As used by ShapedTypeInterface.
        - 'raw' form: uint64_t, where ~0 denotes a dynamic dimension. Used
          when dimensions need to be encoded into MEF ops via attributes.
     - For shapes:
        - 'optional' form: ArrayRef<std::optional<uint64_t>>.
        - 'shaped type' form: ArrayRef<int64_t>, as used by ShapedTypeInterface.
        - 'raw' form: IntArrayElementsAttr of UI64Ints.
     - For dtypes:
        - 'optional' form: std::optional<DType>, where nullopt denotes a
          dynamic dtype.
        - 'shaped type' form: An MLIR type, where NoneType denotes a dynamic
          dtype.
        - 'raw' form: uint8_t, where 0 (ie invalid) denotes a dynamic dtype.

    `!mgp.tensor_specs` are represented by `MLRT::AsyncValueRef<M::TensorSpec>`
    in the host C++ runtime. All dimensions in the runtime value should 'refine'
    the corresponding compile time spec, ie runtime dims must equal statically
    known compile-time dims.

    Examples:
    ```mlir
    // Fully statically known spec.
    !mgp.tensor_spec<4x16xf32>
    // Spec with one dynamic dimension and dynamic dtype.
    !mgp.tensor_spec<?x3xunk>
    ```
  }];

  // NOTE: To support the ShapedTypeInterface::getShape() method we maintain
  // 'optional' (dimsOrEmpty) and 'shaped type' (shapeOrEmpty) forms.
  // See the builders for the smart constructors.
  let parameters = (ins
    ArrayRefParameter<"std::optional<uint64_t>">:$dimsOrEmpty,
    "std::optional<DType>":$optDType,
    ArrayRefParameter<"int64_t">:$shapeOrEmpty
  );

  let assemblyFormat = "`<` custom<DimsAndDType>($dimsOrEmpty, $optDType, $shapeOrEmpty) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// For the ShapedTypeInterface we implement:
    ///   bool hasRank(); // always true.
    ///   ShapedType cloneWith(std::optional<ArrayRef<int64_t>> shape,
    ///                        Type elementType);
    ///   Type getElementType();
    ///   ArrayRef<int64_t> getShape();

    /// Returns the dimensions of this tensor spec.
    llvm::ArrayRef<std::optional<uint64_t>> getDims() const;

    /// Returns the rank of this tensor spec.
    size_t getRank() const;

    /// Returns the number of unknown, aka 'dynamic' dimensions in
    /// this tensor spec.
    size_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or nullopt if the
    /// dimension is dynamic.
    std::optional<uint64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// nullopt if the tensor shape is not fully static (ie at least one
    /// dynamic dimension).
    std::optional<uint64_t> getNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or nullopt if the tensor spec is not fully static
    /// (ie at least one dynamic dimension, or dynamic dtype).
    std::optional<uint64_t> getSizeInBytes() const;

    /// Returns the dimensions of this tensor spec in 'raw form'.
    /// Each element of the result will be an UI64Int, with ~0 denoting
    /// an unknown dimension.
    IntArrayElementsAttr getRawDims() const;

    /// Returns the dtype of this tensor spec in 'raw form'.
    /// The result will be a UI8Int using the DType::value encoding,
    /// with 0 denoting the unknown dtype.
    mlir::IntegerAttr getRawDType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'optional' form.
    TypeBuilder<(ins "llvm::ArrayRef<std::optional<uint64_t>>":$dims,
                     "std::optional<DType>":$optDType)>,

    /// Constructs a tensor spec from its components in 'shaped type' form.
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                        "mlir::Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw form',
    /// using underlying integer values.
    TypeBuilder<(ins "llvm::ArrayRef<uint64_t>":$rawDims, "uint8_t":$rawDType)>,

    /// Construct a tensor spec from its components in 'raw form', using
    /// actual attributes. As a convenience, the null dtype attribute is taken
    /// to denote the unknown element type.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawDims,
                                        "mlir::IntegerAttr":$rawDTypeOrNull)>,
  ];
}

//===----------------------------------------------------------------------===//
//
// Device-generic types in the MGP dialect. All of these types accept an
// #M.device_ref attribute indicating the specific type and 'virtual' device
// id the type should apply to.
//
//===----------------------------------------------------------------------===//

class MGP_DeviceType<string name, string mnemonic, list<Trait> traits = []> :
    MGP_Type<
      name,
      mnemonic,
      [DeclareTypeInterfaceMethods<MGP_HasDeviceTypeInterface>] # traits
    >;

def MGP_OptDeviceRefParam :
  DefaultValuedParameter<"M::DeviceRefAttr",
                         "M::DeviceRefAttr::getDefault($_ctxt)">;

//===----------------------------------------------------------------------===//
// DeviceBufferType
//===----------------------------------------------------------------------===//

def MGP_DeviceBufferType : MGP_DeviceType<"DeviceBuffer", "buffer", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Byte array buffers.";
  let description = [{
    `!mgp.buffer`s are mutable byte arrays which reside on the device given by
    the `#M.device_ref` attribute.

    Nothing is implied about the internal structure of the bytes, such as rank,
    dimensions, underlying element type, padding, or element endianness. The
    compiler is responsible for ensuring the producers and consumers of buffers
    agree on their interpretation of the buffer for the intended device.

    Generally `!mgp.buffer`s are the lowest-common denominator representation
    of tensor data across kernel implementations. Buffers may be wrapped by
    various tensor types, including `!mgp.tensor`, or used directly.

    This type implements `ShapedTypeInterface` as a rank-1 tensor over
    `uint8_t`. This in turn supports the `ElementsAttrInterface` on buffer
    attributes.

    Alignment constraints are specified at buffer creation or in the attribute
    backing a constant buffer, not in the buffer type.

    Since buffers are mutable, any op reading a buffer must accept a
    `!mef.chain`, and any op writing one must return a `!mef.chain`, so the
    compiler can sequence writes and reads.

    `!mgp.buffer`s are represented by (subclasses of)
    `MLRT::AsyncValueRef<BufferRef>` in the host C++ runtime. Some devices,
    such as CUDA, may need to include additional data to control device-side
    synchronization.

    Examples:
    ```mlir
    // A buffer on the CUDA GPU with virtual device id 3.
    !mgp.buffer<<"gpu", 3>>
    // A buffer on the CPU with virtual device id 0.
    !mgp.buffer<<"cpu", 0>>
    // Abbreviation for the above.
    !mgp.buffer
    ```
  }];

  let parameters = (ins MGP_OptDeviceRefParam:$device);
  let assemblyFormat = "(`<` $device^ `>`)?";
  let genVerifyDecl = 1;

  let builders = [
    /// Constructs a buffer type from the given device.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device)>,

    /// Constructs a buffer type which matches the device and any other properties
    /// of the given tensor type.
    TypeBuilderWithInferredContext<(ins "M::MGP::DeviceTensorType":$tensorType)>,
  ];
}

//===----------------------------------------------------------------------===//
// DeviceTensorType
//===----------------------------------------------------------------------===//

def MGP_DeviceTensorType : MGP_DeviceType<"DeviceTensor", "tensor", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Type of tensors respecting a tensor spec.";
  let description = [{
    `!mgp.tensor`s are tensors consisting of a buffer and a tensor spec, where
    the buffer resides on the device given by the `#M.device_ref` attribute.
    The elements in the buffer are generally in dense row-major order
    according to the runtime tensor spec, with endianness according to the
    device's architecture.

    Tensors are immutable unless marked `mut`. When an immutable tensor is
    created from a buffer, the buffer must be copied if it could still be
    written. When a buffer is extracted from an immutable tensor, the buffer
    must be copied if the tensor could still be read.

    The `mut` marker is the lowered form of `!mo.buffer`: a tensor the model
    writes in place through explicitly chained ops (a KV cache, for example).

    The `stable` marker indicates that the tensor's underlying address is
    stable across dispatches and may participate in cross-device coordination
    mechanisms such as `!mgp.stable_addr` slots. A tensor may be both `mut` and
    `stable`.

    `!mgp.tensor`s are represented by (subclasses of)
    `MLRT::AsyncValueRef<Tensor>` in the host C++ runtime, and the underlying
    buffer may also be passed as a raw `void*`. Some devices, such as CUDA, may
    need additional data to control device-side synchronization.

    Examples:
    ```mlir
    // Statically-shaped tensor on the CUDA GPU with virtual device id 3.
    !mgp.tensor<4x16xf32, <"gpu", 3>>
    // Dynamically-shaped tensor on the CPU with virtual device id 0.
    !mgp.tensor<?x?xf32, <"cpu", 0>>
    // Abbreviation for the above.
    !mgp.tensor<?x?xf32>
    // A mutable tensor (lowered from !mo.buffer) on GPU 0.
    !mgp.tensor<464x4800xf16, <"gpu", 0>, mut>
    // A stable tensor on GPU 0.
    !mgp.tensor<4x16xf32, <"gpu", 0>, stable>
    // A mutable, stable tensor on GPU 0.
    !mgp.tensor<4x16xf32, <"gpu", 0>, mut, stable>
    ```
  }];

  let parameters = (ins
    MGP_OptDeviceRefParam:$device,
    "M::MGP::TensorSpecType":$tensorSpec,
    DefaultValuedParameter<"bool", "false">:$mut,
    DefaultValuedParameter<"bool", "false">:$stable
  );

  // The device, mut, and stable tail share one custom directive: the optional
  // groups would otherwise start with `,`, which the generated parser cannot
  // disambiguate.
  let assemblyFormat = [{
    `<` custom<TensorSpec>($tensorSpec) `` custom<DeviceTensorTail>($device, $mut, $stable) `>`
  }];

  let builders = [
    /// Constructs an immutable, non-stable tensor type from the given device
    /// and tensor spec.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "M::MGP::TensorSpecType":$tensorSpecType)>,

    /// Constructs a tensor type from the given device, tensor spec, and
    /// mutability. The result is not stable.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "M::MGP::TensorSpecType":$tensorSpecType,
                                        "bool":$mut)>,

    /// Constructs a tensor type from the given device, tensor spec,
    /// mutability, and stability.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "M::MGP::TensorSpecType":$tensorSpecType,
                                        "bool":$mut,
                                        "bool":$stable)>,
  ];
}

//===----------------------------------------------------------------------===//
// StableAddressType
//===----------------------------------------------------------------------===//

def MGP_StableAddressType : MGP_Type<"StableAddress", "stable_addr"> {
  let summary = "Handle to a stable address slot for cross-device coordination.";
  let description = [{
    Inside a device-graph body, `mgp.device_graph.stable.place` publishes a
    value into the slot and `mgp.device_graph.stable.get` reads it; both return
    the slot. `mgp.device_graph.stable.tensor` views the slot as the
    `!mgp.tensor<..., stable>` it holds, and a kernel handed that tensor reads
    through the slot via a `stable`-marked `!mo.tensor`.

    This type is the MGP-level handle into the low-level allocation and
    pointer-patching mechanism.

    The `elementType` parameter records the `!mgp.tensor` type held by the
    slot, so a view built over the slot knows its shape and device.
  }];
  let parameters = (ins MGP_DeviceTensorType:$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    /// Builds a `!mgp.stable_addr<T>` without an explicit context, reusing the
    /// context of the wrapped tensor type.
    TypeBuilderWithInferredContext<(ins "Type":$tensorType), [{
      return $_get(tensorType.getContext(),
                    llvm::cast<DeviceTensorType>(tensorType));
    }]>,
  ];
}

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
