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
// This defines operations in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPS_TD
#define GENERICML_MGPDIALECT_MGPOPS_TD

include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.td"
include "GraphCompiler/MGPDialect/Dialect/MGPAttrs.td"
include "GraphCompiler/MGPDialect/Dialect/MGPDeviceGraphOps.td"
include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypes.td"

include "GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "MLRT/GraphRT/GRTDialect/GRTChainInterface.td"
include "MLRT/GraphRT/GRTDialect/GRTExtendsLifetimeInterface.td"
include "MLRT/GraphRT/GRTDialect/GRTMayAliasInterface.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/POPDialect/POPTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class MGP_Op<string mnemonic, list<Trait> traits = []> :
  Op<MGPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def MGP_ResourceConstantOp : MGP_Op<"resource.constant", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let arguments = (ins IndexAttr:$resourceSlot);
  let results = (outs MGP_DeviceBufferType:$outBuffer);

  let summary = "Retrieves a pointer to the corresponding constant resources";
  let description = [{
    Retrieves a pointer to the constant resources attached to the parent
    `mogg.func`. Kernels differing only in constant values still merge
    when the index matches, which speeds up compilation.
  }];

  let assemblyFormat = [{
    `(` operands `)` `resourceSlot` `=` $resourceSlot attr-dict
        `:` functional-type(operands, results)
  }];
}

def BufferConstantOp : MGP_Op<"buffer.constant", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
    "getLiftableAttributes",
    "getIgnorableAttributes"
  ]>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Allocates and initializes a buffer from a host attribute.";
  let description = [{
    Returns an `!mgp.buffer` initialized with the byte array in the given attribute
    (which will be loaded from the MEFFile holding the compiled model).

    Example:
    ```mlir
      %buf = mgp.buffer.constant
               dense_resource<label>:!M.aligned_bytes<256, align 16>
    ```
  }];

  let arguments = (ins
    ElementsAttr:$value,
    UI64Attr:$rawAlign
  );
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    custom<ElementsAttrWithAlign>($value, $rawAlign) attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let builders = [
    OpBuilder<(ins "mlir::ElementsAttr":$value)>
  ];

  let extraClassDefinition = [{
    bool BufferConstantOp::isTrivial() { return true; }

    SmallVector<Attribute> BufferConstantOp::getLiftableAttributes() {
      return {getValueAttr()};
    }

    SmallVector<Attribute> BufferConstantOp::getIgnorableAttributes() {
      return {getRawAlignAttr()};
    }

  }];

}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MGP_Op<"context.create"> {
  let summary = "Creates a context with given number of state slots.";
  let description = [{
    Creates a context with the given number of 'state slots'.

    Example:
    ```mlir
    %ctx = mgp.context.create 4
    %ctx : !mgp.context
    ```
  }];

  let arguments = (ins UI64Attr:$numSlots);
  let results = (outs MGP_ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}


//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

def TensorCreateOp : MGP_Op<"tensor.create", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  DeclareOpInterfaceMethods<MGP_CreateTensorOpInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Creates an `!mgp.tensor` from an `!mgp.tensor_spec` and "
                "`!mgp.buffer`.";
  let description = [{
    If this op holds the only reference to the buffer then it is moved into
    the tensor. Otherwise a deep copy of the buffer is taken.

    Example:
    ```mlir
    %ctx : !mef.context
    %ch : !mef.chain
    %buf : !mgp.buffer<<"gpu", 3>>
    %spec : !mgp.tensor_spec<4x16xf32>
    %tensor = mgp.tensor.create %ch, %ctx[5], %buf, %spec : !mgp.tensor<4x16xf32, <"gpu", 3>>
    %tensor : !mgp.tensor<4x16xf32, <"gpu", 3>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer,
    MGP_TensorSpecType:$tensorSpec,
    // Mirrors the `mut` marker on the result type: the type parameter is not
    // derivable from the operands, allowing the use of InferTypeOpAdaptor.
    UnitAttr:$mut,
    // Mirrors the `stable` marker on the result type.
    UnitAttr:$stable
  );
  let results = (outs MGP_DeviceTensorType:$outTensor);
  let assemblyFormat = [{
    $inChain `,` $buffer `,` $tensorSpec
    (`mut` $mut^)? (`stable` $stable^)? attr-dict
    `:` custom<TensorTypeToDeviceBufferSpec>(type($outTensor),
          type($buffer), type($tensorSpec), type($inChain))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

def TensorExtractBufferOp : MGP_Op<"tensor.extract.buffer", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Extracts the `!mgp.buffer` from an `!mgp.tensor`.";
  let description = [{
    If this op holds the only reference to the tensor then the buffer will be
    moved out of the tensor. Otherwise a deep copy of the buffer is returned.

    Example:
    ```mlir
    %tensor : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %buf = mgp.tensor.extract.buffer %tensor : <8x64xf32, <"gpu", 3>>
    %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins MGP_DeviceTensorType:$tensor);
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $tensor attr-dict `:` type($tensor)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//

def TensorExtractTensorSpecOp : MGP_Op<"tensor.extract.tensor_spec", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Extracts the `!mgp.tensor_spec` from a `!mgp.tensor`.";
  let description = [{
    Returns the `!mgp.tensor_spec` of a tensor. If requested, the dtype
    can be replaced with `unk` to represent the tensor's 'shape'.

    Example:
    ```mlir
    %tensor : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %shape0 = mgp.tensor.extract.tensor_spec %tensor : <8x64xf32, <"gpu", 3>>
    %shape0 : !mgp.tensor_spec<8x64xk32>
    %shape1 = mgp.tensor.extract.tensor_spec %tensor : <8x64xf32, <"gpu", 3>>
    %shape1 : !mgp.tensor_spec<8x64xunk>
    ```
  }];

  let arguments = (ins MGP_DeviceTensorType:$tensor);
  let results = (outs MGP_TensorSpecType:$outSpec);
  let assemblyFormat = [{
    $tensor attr-dict `:` type($tensor)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TensorSliceOp
//===----------------------------------------------------------------------===//

def TensorSliceOp : MGP_Op<"tensor.slice", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Zero-copy contiguous slice of a tensor along one dimension.";
  let description = [{
    Returns a view into the input tensor starting at the given offset. No data
    is copied; the output tensor borrows from the input. The sliced dimension
    k, the stride in elements, and the start offset are all computed at runtime
    inside the Mojo primitive from the input shape and the start buffer.

    The input tensor may reside on any device. The start tensor must be on the
    host (cpu) since it is metadata; no device data is read from it.

    The device context must be on the same device as the input; backends whose
    runtime only accepts registered pointers register the slice through it.

    Example:
    ```mlir
    %input : !mgp.tensor<5xf32, <"gpu", 0>>
    %spec  : !mgp.tensor_spec<3xf32>
    %start : !mgp.tensor<1xsi64, <"cpu", 0>>
    %ctx   : !mgp.device_context<<"gpu", 0>>
    %out = mgp.tensor.slice %input, %spec, %start, %ctx
             : !mgp.tensor<5xf32, <"gpu", 0>>, !mgp.tensor_spec<3xf32>,
               !mgp.tensor<1xsi64, <"cpu", 0>>,
               !mgp.device_context<<"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceTensorType:$input,
    MGP_TensorSpecType:$outputSpec,
    MGP_DeviceTensorType:$start,
    MGP_DeviceContextType:$deviceContext
  );

  let results = (outs MGP_DeviceTensorType:$output);

  let assemblyFormat = [{
    $input `,` $outputSpec `,` $start `,` $deviceContext attr-dict `:`
    type($input) `,` type($outputSpec) `,` type($start) `,`
    qualified(type($deviceContext))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorSpecCreateOp
//===----------------------------------------------------------------------===//

def TensorSpecCreateOp : MGP_Op<"tensor_spec.create", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = [{
    Creates a CPU tensor spec from a mix of runtime and static dimensions and
    a statically known dtype.
  ]};
  let description = [{
    Creates a CPU tensor spec of statically known rank with dimensions taken
    from either runtime operands or statically known attributes. The element
    type must be known statically. One operand is given per unknown dimension,
    in order (none if all dimensions are static).

    Tensor specs are represented as M::TensorSpecs in the Modular runtime.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
    %spec : !mgp.tensor_spec<?x?xf32>

    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<32x4xf32>
    %spec : !mgp.tensor_spec<32x4xf32>
    ```
  }];

  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by MEF.
  let arguments = (ins
    Variadic<Index>:$runtimeDims,
    UI64ArrayElementsAttr:$rawDims,
    UI8Attr:$rawDType
  );
  let results = (outs MGP_TensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims attr-dict
    `:` custom<RawDimsAndDType>($rawDims, $rawDType)
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "TensorSpecType":$spec,
      CArg<"ValueRange", "{}">:$runtimeDims
    )>
  ];
}

//===----------------------------------------------------------------------===//
// TensorSpecGetDimOp
//===----------------------------------------------------------------------===//

def TensorSpecGetDimOp : MGP_Op<"tensor_spec.get_dim", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  Pure,
]> {
  let summary = [{
    Return the dimension of the tensor_spec at the specified axis.
  ]};
  let description = [{
    Returns the actual runtime value of each dimension in a CPU tensor spec.
    If the dimension is known statically, it is returned as a constant index.
  }];

  let arguments = (ins
    MGP_TensorSpecType:$spec,
    UI64Attr:$axis
  );
  let results = (outs Index:$dim);
  let assemblyFormat = [{
    `[` $axis `]` $spec attr-dict `:` type($spec)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : MGP_Op<"assert", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getLiftableAttributes"]>,
]> {
  let summary = "Assert that a boolean condition is true.";
  let description = [{
    Return an error with the given message if the condition is false.

    Example:
    ```mlir
    %ch = mgp.assert %cond, "condition is not true"
    ```
  }];

  let arguments = (ins I1:$cond, StrAttr:$message);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = "$cond `,` $message attr-dict";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ExecuteGenericOp
//===----------------------------------------------------------------------===//

def ExecuteGenericOp : MGP_Op<"generic.execute"> {
  let summary = "Executes a function loaded into the CPU Runtime";
  let description = [{
    Executes a function already loaded into the compiled kernel context at the
    given slot/module index within the `!mgp.context`, using the generic API:
    the primitive passes raw async values for the function to unpack.

    Always returns an output chain in addition to the kernel results, for error
    handling or side-effect dependencies.
  }];

  let arguments = (ins
    MGP_ContextType:$theContext,
    Variadic<AnyType>:$args,
    UI64Attr:$aRuntimeSlot,  // 0
    UI64Attr:$bModuleIndex,  // 1
    UI64Attr:$cFunctionIndex,  // 2
    StrAttr:$dSummary,  // 3
    DefaultValuedAttr<BoolAttr, "false">:$eIsAsync,  // 4
    ArrayAttr:$fConstantResource  // 5
  );
  let results = (outs Variadic<AnyType>:$outResults);
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

def GuardOp : MGP_Op<"guard", [
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
]> {
  let summary = "SSA-value guard operation";

  let description = [{
    Consumes a chain and a variadic list of values and forwards the non-chain
    inputs, establishing execution dependencies between the chain-producing op
    and downstream operations.

    Example:

    ```mlir
      %X: !mgp.tensor<?x?xf32>
      %Y: !mgp.tensor<?x?xf32>

      %spec1_1 = mgp.tensor_spec.get_dim[1] %0 : !mgp.tensor_spec<?x?xf32>
      %spec2_0 = mgp.tensor_spec.get_dim[0] %2 : !mgp.tensor_spec<?x?xf32>

      %cond = index.cmp.eq %spec1_1, %spec2_0

      %chain = mgp.assert %cond, "D1 != D2"
      %Xp, %Yp = mgp.guard %chain, %X, %Y : !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>
    ```
  }];

  let arguments = (ins
    ChainType:$chain,
    Variadic<AnyType>:$inputs
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` qualified(type($inputs))
  }];

  let hasVerifier = 1;
}


def ExtendLifetimeOp : MGP_Op<"extend.lifetime", [
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "SSA-value guard operation";

  let description = [{
    Returns a chain that is ready when all inputs are, extending the lifetime
    of the input values to match the lifetime of the input chain.
  }];

  let arguments = (ins
    ChainType:$gate,
    Variadic<AnyType>:$inputs
  );
  let results = (outs ChainType:$results);
  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelFrameworkAttr : ConfinedAttr<StrAttr, [
  AttrConstraint<
    CPred<"CompiledFrameworkLabel::isValidFrameworkName("
          "cast<StringAttr>($_self).getValue())">,
    "must be a recognized framework name.">,
]>;

def ModelOp : MGP_Op<"model", [
  HasOnlyGraphRegion,
  HasParent<"ModuleOp">,
  IsolatedFromAbove,
  RegionKindInterface,
  SymbolName,
  SymbolVisibility,
  Symbol,
]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    The 'init' and 'execute' entry points are single-block regions.

    The 'init' entry point has the implicit signature:
      (!mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, ...) ->
      (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, ...)
    where:
      mgp.opaques are the weights registry and device contexts.

    The 'execute' entry point has the implicit signature:
      (!mef.chain, !mgp.context, weights_registry, dev0, ... devn, in1, ..., inn) ->
      (!mef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is the op's explicit signature, and weights_registry and dev0, ..., devn
    are those passed to init.

    The Modular runtime implicitly calls 'init' when the model executable is
    first loaded, and re-uses the resulting context value for each subsequent
    call to 'execute'.

    This op is represented by a subclass of M::Model in the Modular runtime,
    chosen by the `frameworkName` attribute, which tools inspect to decide
    which API mediates between framework-specific values and the model's.

    Additional attributes may be supplied; these names have special meaning:
     - argument_names
     - result_names

    Example:
    ```mlir
    // At the top-level
    mgp.model @example_model : mof { attr="value" }
      init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
        %ch = mef.chain.new
        %ctx = mgp.context.create 0
        mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque, !mgp.opaque
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        mef.output %ch, %arg : !mef.chain, tensor<?x4x16xf32>
      }
    ```
  }];

  // CAUTION: Constraints on these attribute names and their order:
  //  - 'sym_name' must match mlir::SymbolTable::getSymbolAttrName().
  //  - The extra attributes must appear last since optional.
  let arguments = (ins
    M_DeviceRefArrayAttr:$devices,
    ModelFrameworkAttr:$frameworkName,
    TypeAttrOf<FunctionType>:$inputOutputTypes,
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    UnitAttr:$isSubgraph,
    UnitAttr:$isDeviceGraph,
    OptionalAttr<DictionaryAttr>:$theExtraAttrs
  );

  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }

    /// Required by the internal callgraph utilities.
    Region &getBodyRegion() { return getExecute(); }

    MutableArrayRef<BlockArgument> getInitDevices() {
      return getInitBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitInitArgs).take_front(getDevices().size());
    }
    OperandRange getInitOutputDevices() {
      return getInitBlock()->getTerminator()->getOperands().drop_front(
        // Fixed implicit execute args are also fixed implicit init output args.
        kMgpModelNumFixedImplicitExecArgs
      ).take_front(getDevices().size());
    }

    BlockArgument getInitWeightsRegistry() {
      return getInitBlock()->getArgument(0);
    }

    MutableArrayRef<BlockArgument> getExecuteInputs() {
      return getExecuteBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitExecArgs + getDevices().size());
    }
    MutableArrayRef<BlockArgument> getExecuteChainContext() {
      return getExecuteBlock()->getArguments().take_front(2);
    }
    MutableArrayRef<BlockArgument> getExecuteDevices() {
      return getExecuteBlock()->getArguments().drop_front(kMgpModelNumFixedImplicitExecArgs).take_front(getDevices().size());
    }
    MutableArrayRef<BlockArgument> getExecuteImplicitArguments() {
      return getExecuteBlock()->getArguments().take_front(kMgpModelNumFixedImplicitExecArgs + getDevices().size());
    }
    BlockArgument getExecuteChain() { return getExecuteBlock()->getArgument(0); }
    BlockArgument getExecuteWeightsRegistry() { return getExecuteBlock()->getArgument(2); }
    BlockArgument getSubgraphIndex() { return getExecuteBlock()->getArguments().back(); }

    /// Retrieve the single ContextCreateOp from the init block.
    ContextCreateOp getInitContext();

    /// Retrieve the context for the execute block.
    TypedValue<MGP::ContextType> getExecuteContext();


    StringAttr getName() { return getSymNameAttr(); }
  }];

  let builders = [
    /// Creates a model with empty 'init' and 'execute' bodies and explicit
    /// signature `inputTypes` -> `outputTypes`. `inputArgLocs` gives the
    /// locations of the explicit 'execute' arguments, one per input type.
    OpBuilder<(ins
      "StringAttr":$sym_name,
      "StringRef":$frameworkName,
      "ArrayRef<DeviceRef>":$devices,
      "ArrayRef<Type>":$inputTypes,
      "ArrayRef<Type>":$outputTypes,
      "ArrayRef<Location>":$inputArgLocs,
      CArg<"bool", "false">:$isSubgraph,
      CArg<"bool", "false">:$isDeviceGraph,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs
    )>
  ];

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) `` $sym_name
      (`subgraph` $isSubgraph^)? (`device_graph` $isDeviceGraph^)?
      `:` custom<ClassName>($frameworkName)
      ($theExtraAttrs^)?
      `devices` $devices
      `init` custom<InitRegion>($init)
      `execute` custom<ExecuteRegion>($inputOutputTypes, $execute, ref($devices))
      attr-dict
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RuntimeCreateOp
//===----------------------------------------------------------------------===//

def RuntimeCreateOp : MGP_Op<"runtime.create", [
  DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,
]> {
  let summary = "Creates a runtime to coordinate execution.";
  let description = [{
    Creates a runtime within an `!mgp.context` slot.

    Note that this operation accepts and returns `!mef.chain`s since it is
    part of device setup on the host.

    Example:
    ```mlir
      // Establish a runtime in slot 5.
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %ch1 = mgp.runtime.create %ch0, %ctx[5]
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    UI64Attr:$aRuntimeSlot  // 1
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` attr-dict
  }];

  let extraClassDefinition = [{
    uint64_t RuntimeCreateOp::getSlot() { return getARuntimeSlot(); }

  }];
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

def BufferGetCachedOp : MGP_Op<"buffer.get_cached", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Retrieves an `!mgp.buffer` from a dedicated state slot.";
  let description = [{
    Returns (a borrow of) the `!mgp.buffer` in the state slot with the given
    index. The buffer should have already been cached by `mgp.buffer.cache`.
    Returns an error if the cached buffer's size (if any) or device do not
    agree with the `!mgp.buffer` type annotation.

    Note this op is gated by a `!mef.chain` since the cache is held on the
    host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch0 : !mef.chain
    %idx : index
    // Returns the buffer cached in slot index.
    %buf = mgp.buffer.get_cached %ch0, %ctx[%idx] : !mgp.buffer<<"gpu", 3>>
    %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    Index:$bBufferSlot  // 1
  );

  let results = (outs
    MGP_DeviceBufferType:$outBuffer,
    Optional<PointerOf<PointerOf<KGEN_NoneType>>>:$optional_memStorageHandle
  );

  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bBufferSlot `]` attr-dict
    `:` qualified(type($outBuffer)) (`,` qualified(type($optional_memStorageHandle))^)?
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// IntCacheOp
//===----------------------------------------------------------------------===//

def IntCacheOp : MGP_Op<"int.cache", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Stores an integer value into a dedicated state slot.";
  let description = [{
    Stores the given integer value into the state slot at the given index.
    It may be later retrieved by `mgp.int.get_cached`.

    Example:
    ```mlir
    %ctx : !mgp.context
    %val : index
    %ch0 : !mef.chain
    // Cache the integer in slot 42.
    %ch1 = mgp.int.cache %ch0, %ctx[42], %val
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    Index:$value,
    UI64Attr:$bIntSlot
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bIntSlot `]` `,` $value attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// IntGetCachedOp
//===----------------------------------------------------------------------===//

def IntGetCachedOp : MGP_Op<"int.get_cached", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Retrieves an integer value from a dedicated state slot.";
  let description = [{
    Returns the integer value stored in the state slot at the given index.
    The value should have already been stored by `mgp.int.cache`.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch0 : !mef.chain
    %idx : index
    %val = mgp.int.get_cached %ch0, %ctx[%idx]
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    Index:$bIntSlot
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bIntSlot `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BufferGetSizeOp
//===----------------------------------------------------------------------===//

def BufferGetSizeOp : MGP_Op<"buffer.get_size", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Get the runtime size of a buffer in bytes.";
  let description = [{
    Returns the size of the input buffer in bytes.

    Example:
    ```mlir
    %in_buf : !mgp.buffer
    %buf_size = mgp.buffer.get_size %in_buf
    %buf_size : index
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer
  );
  let results = (outs Index:$size);
  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
    `:` qualified(type($buffer))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

def DebugPrintOp : MGP_Op<"debug.print", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Prints a debug string. For debugging and testing only.";
  let description = [{
    If a label attribute is supplied the string is printed with that label.

    Note this op accepts and returns `!mef.chain`s since printing is managed
    by the host.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %ch1 = mgp.debug.print %ch0, %ctx, "debug string", "label"
    %ch1: !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    StrAttr:$debugString,
    DefaultValuedAttr<StrAttr, "\"\"">:$label
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `,` $debugString ( `,` $label^ )? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DebugPrintIntOp
//===----------------------------------------------------------------------===//

def DebugPrintIntOp : MGP_Op<"debug.print.int", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Prints a runtime integer value. For debugging and testing only.";
  let description = [{
    Prints a runtime index value. If a label attribute is supplied the value
    is printed with that label.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %val: index
    %ch1 = mgp.debug.print.int %ch0, %ctx, %val, "label"
    %ch1: !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    Index:$value,
    DefaultValuedAttr<StrAttr, "\"\"">:$aLabel
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `,` $value ( `,` $aLabel^ )? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def MGP_CallOp : MGP_Op<"call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Call a subgraph.";
  let description = [{
    Placeholder for the execute block of the !mgp.model named by `callee`,
    replaced by mef.callables and grt.invoke ops by
    `ConvertSubgraphsToCallablesPass`.

    Operands and results must match the signature of the target model's execute block.

    Example:
    ```mlir
    mgp.model @subgraph : mof {
      init (...) {
        // ...
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        // ...
      }
    }

    mgp.model @main : mof {
      init (...) {
        // ...
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        %sub_res = mgp.call @subgraph(%ch, %ctx, %weights, %cpu, %arg)
      }
    }
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    DefaultValuedAttr<StrAttr, "\"\"">:$prefix,
    Variadic<AnyType>:$operands,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isGlobalizedIndex,

    // Needed to satisfy CallOpInterface
    DefaultValuedOptionalAttr<DictArrayAttr, "{}">:$arg_attrs,
    DefaultValuedOptionalAttr<DictArrayAttr, "{}">:$res_attrs
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $operands `)`
    (`arg_attrs` `=` $arg_attrs^)?
    (`isGlobalizedIndex` `=` $isGlobalizedIndex^)?
    (`prefix` `=` $prefix^)?
    (`res_attrs` `=` $res_attrs^)?
    attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return getOperands();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    FlatSymbolRefAttr getCalleeSymbol() {
      return cast<FlatSymbolRefAttr>(getCalleeAttr());
    }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      this->setCalleeAttr(cast<FlatSymbolRefAttr>(cast<SymbolRefAttr>(callee)));
    }
  }];
}

//===----------------------------------------------------------------------===//
// StringCreateOp
//===----------------------------------------------------------------------===//

def StringCreateOp : MGP_Op<"string.create", [Pure]> {
  let summary = "Creates a Mojo String from a string attribute.";
  let description = [{
    Returns an opaque pointer to a Mojo `String`, created from a Mojo
    `Pointer[String]`.

    Example:
    ```mlir
    %str = mgp.string.create "abcd" : !mgp.opaque<"mojo_string">
    ```
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs MGP_OpaqueType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// StringConcatOp
//===----------------------------------------------------------------------===//

def StringConcatOp : MGP_Op<"string.concat", [Pure]> {
  let summary = "Concatenates two Mojo Strings.";
  let description = [{
    Returns a new Mojo String. Both inputs must be opaque pointers to Mojo
    `String` values (as created by `mgp.string.create`).

    Example:
    ```mlir
    %prefix = mgp.string.create "hello " : !mgp.opaque<"mojo_string">
    %end = mgp.string.create "world" : !mgp.opaque<"mojo_string">
    %result = mgp.string.concat %prefix, %end : (!mgp.opaque<"mojo_string">, !mgp.opaque<"mojo_string">) -> !mgp.opaque<"mojo_string">
    // %result contains "hello world"
    ```
  }];

  let arguments = (ins MGP_OpaqueType:$prefix, MGP_OpaqueType:$end);
  let results = (outs MGP_OpaqueType:$result);
  let assemblyFormat = [{
    $prefix `,` $end attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
//
// Device-generic operations in the MGP dialect. Most accept (at least one)
// #M.device_ref attribute giving the device type and 'virtual' id. Ops that
// also reference the 'host' device implement `MGP_HasDeviceOpInterface`.
//
// `DevicePrimitivesPass` renames ops implementing `MGP_HasDeviceOpInterface`
// with their device's label as a suffix and replaces #M.device_ref attributes
// with the device id. This keeps first-class device attributes for lowering
// and MGP-level transformations while dispatching directly to per-device
// primitives without handling labels at runtime.
//
// Chains in MGP are plain `!mef.chain` values.
//
//===----------------------------------------------------------------------===//

class MGP_DeviceOp<string mnemonic, list<Trait> traits = []> :
  MGP_Op<mnemonic, [
    DeclareOpInterfaceMethods<MGP_HasDeviceOpInterface>,
  ] # traits>;

def MGP_OptDeviceRefAttr :
  DefaultValuedAttr<M_DeviceRefAttr, "kCPULabel, 0">;

//===----------------------------------------------------------------------===//
// DeviceContextProfileStartOp
//===----------------------------------------------------------------------===//

def DeviceContextProfileStartOp : MGP_DeviceOp<"device.context.profile.start"> {
  let summary = "Shim to tag the start of a device context profiling run";
  let description = [{Temporary op for onnx profiling; should use max-driver instead.}];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceContextType:$deviceContext,
    MGP_OptDeviceRefAttr:$bDevice,  // 1
    StrAttr:$cTag,
    StrAttr:$dFilePath  // 2
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` ( `device` $bDevice^ )?
      `cTag` `=` $cTag `,` `dFilePath` `=` $dFilePath
      attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{

    M::DeviceRefAttr DeviceContextProfileStartOp::getDevice() { return getBDevice(); }
  }];
}

def DeviceContextProfileEndOp : MGP_DeviceOp<"device.context.profile.end"> {
  let summary = "Shim to tag the end of a device context profiling run";
  let description = [{Temporary op for onnx profiling; should use max-driver instead.}];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceContextType:$deviceContext,
    MGP_OptDeviceRefAttr:$bDevice,  // 1
    StrAttr:$cTag,
    StrAttr:$dFilePath  // 2
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` ( `device` $bDevice^ )?
      `cTag` `=` $cTag `,` `dFilePath` `=` $dFilePath
      attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{

    M::DeviceRefAttr DeviceContextProfileEndOp::getDevice() { return getBDevice(); }
  }];
}


//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

def ModuleLoadOp : MGP_DeviceOp<"module.load", [
  DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,
]> {
  let summary = "Loads functions and constants for device from compiled module.";
  let description = [{
    Loads functions and constants from a compiled module encoded into a byte
    array (or, for unit testing only, a multi-line string), associated with the
    runtime in the given `!mgp.context` slot. The module gets the device's next
    module index, and each function and constant the next index within it (all
    starting at 0). Functions are referenced by (module index, function index)
    in `mgp.generic.execute`, constants by (module index, constant index) in
    `mgp.buffer.constant.load`.

    Each device type may use its own module representation in the host runtime.

    Note that this operation accepts and returns `!mef.chain`s since it is
    part of device setup on the host.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      // Load a CUDA PTX or cubin module with given exported kernel and
      // constant names. The CUDA module will be associated with the CUDA
      // runtime established in slot 5, which should be for the CUDA GPU with
      // virtual device id 3.
      %ch1 = mgp.module.load %ch0, %ctx[5][1],
               dense_resource<label> : !M.aligned_bytes<1000, align 16>,
               ["kernel1", "kernel2"], ["const1", "const2"]
               device <"gpu", 3>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    UI64Attr:$aRuntimeSlot,  // 0
    AnyAttrOf<[M_MultiLineStringAttr, ElementsAttr]>:$bImage,  // 1
    M_MultiLineStringAttr:$cFunctionSymbols,  // 2
    M_MultiLineStringAttr:$dConstantSymbols,  // 3
    MGP_OptDeviceRefAttr:$eDevice,  // 4
    UI64Attr:$fModuleIndex,  // 5
    DefaultValuedAttr<BoolAttr, "false">:$gDebuginfo  // 6
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `[` $fModuleIndex `]` `,` $bImage `,`
    $cFunctionSymbols `,` $dConstantSymbols
    ( `device` $eDevice^ )? ( `debuginfo` $gDebuginfo^ )? attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ModuleLoadOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr ModuleLoadOp::getDevice() { return getEDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// ModuleLoadPlaceholder
//===----------------------------------------------------------------------===//

def ModuleLoadPlaceholderOp : MGP_Op<"module.load.placeholder"> {
  let summary = "Placeholder for the module load";
  let description = [{
    Provides a chain linking the compiled execute to the module load op in
    init. Added just before lowering the kernel ABI, and removed once the load
    module op is created.
  }];
  let results = (outs ChainType:$outChain);
}

//===----------------------------------------------------------------------===//
// BufferConstantExternalOp
//===----------------------------------------------------------------------===//

def BufferConstantExternalOp : MGP_DeviceOp<"buffer.constant.external", [
  DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
    "getLiftableAttributes",
    "getIgnorableAttributes"
  ]>,
  InferTypeOpAdaptor,
]> {
  let summary = "Initializes a buffer from an external constant.";
  let description = [{
    Returns an `!mgp.buffer` on the given device, backed at runtime by the
    pointer a weights registry (a map from names to pointers) holds for `name`.
    The name is an `!mgp.opaque<"mojo_string">`, typically from
    `mgp.string.create` (and optionally `mgp.string.concat` for prefixes).
    The pointer must respect the op's alignment. Like mgp.buffer.constant, the
    buffer's lifetime is untracked and assumed to outlive graph execution.

    Example:
    ```mlir
    %weights : !mgp.opaque<"weights_registry">
    %ch : !mef.chain
    %name = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
    // Return a buffer corresponding to the named weight "foo" in the
    // weights registry.
    %buf = mgp.buffer.constant.external %ch, %weights[%name, 16384, align 16]
      : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_OpaqueType:$weightsRegistry,
    MGP_OpaqueType:$name,
    UI64Attr:$aSize,  // 0
    UI64Attr:$bAlign,  // 1
    MGP_OptDeviceRefAttr:$cDevice,  // 2
    DefaultValuedAttr<BoolAttr, "false">:$dHasAlias,  // 3
    DefaultValuedAttr<BoolAttr, "false">:$eIsPlaceholder  // 4
  );

  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $weightsRegistry
    `[`
    $name `,` $aSize `,` `align` $bAlign
    oilist( `has_alias` $dHasAlias
          | `placeholder` $eIsPlaceholder
          | `device` $cDevice)
    `]`
    attr-dict `:` qualified(type($weightsRegistry)) `,` qualified(type($name))
    `->` qualified(type($outBuffer))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let extraClassDefinition = [{
    SmallVector<Attribute> BufferConstantExternalOp::getLiftableAttributes() {
      return {getASizeAttr(), getBAlignAttr()};
    }

    SmallVector<Attribute> BufferConstantExternalOp::getIgnorableAttributes() {
      return {getCDeviceAttr(), getDHasAliasAttr()};
    }

    M::DeviceRefAttr BufferConstantExternalOp::getDevice() { return getCDevice(); }

    bool BufferConstantExternalOp::isTrivial() { return false; }
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//

def BufferAllocOp : MGP_DeviceOp<"buffer.alloc", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
  DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getIgnorableAttributes"]>,
  InferTypeOpAdaptor,
]> {
  let summary = "Allocates a buffer with dynamically provided byte size on "
                "given device.";
  let description = [{
    Allocates (possibly asynchronously) an uninitialized buffer from a device's
    default memory pool with given byte size, and optional statically known
    alignment. The allocation uses the device-appropriate runtime established
    in the given `!mgp.context` slot. Not all devices support the notion of
    alignment.

    The buffer will be freed when the last reference to it is destroyed.

    Example:
    ```mlir
      %ch : !mef.chain
      %ctx : !mgp.context
      %size : index
      // Allocates %size bytes using the CUDA runtime from slot 5, which
      // should correspond to the CUDA GPU with virtual device id 3.
      %buf = mgp.buffer.alloc %ch, %size : !mgp.buffer<<"gpu", 3>>
      %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    Index:$runtimeSize,
    MGP_DeviceContextType:$deviceContext,
    DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign,
    MGP_OptDeviceRefAttr:$device
  );
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain  `,` $runtimeSize `,` $deviceContext
    ( `,` `align` $rawAlign^ )? attr-dict
    `:` custom<BufferTypeToDevice>(
          type($outBuffer), $device, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    SmallVector<mlir::Attribute> BufferAllocOp::getIgnorableAttributes() {
      return {getRawAlignAttr(), getDeviceAttr()};
    }
  }];
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

def BufferCacheOp : MGP_Op<"buffer.cache"> {
  let summary = "Moves an `!mgp.buffer` into a dedicated state slot.";
  let description = [{
    Stores the given `!mgp.buffer` into the state slot at the given index.
    It may be later retrieved by `mgp.buffer.get_cached`. Depending on the
    buffer's device, any computation associated with the buffer may need to be
    'finished' before the store is executed.

    If the buffer is immutable, or this op holds the only reference to the
    buffer, then it will be moved into the cache. Otherwise a deep copy of
    the buffer will be taken.

    Note this op accepts and returns `!mef.chain`s since the buffer cache is
    held on the host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf : !mgp.buffer<<"gpu", 3>>
    %ch0 : !mef.chain
    // Cache the given buffer in slot index 42.
    %ch1 = mgp.buffer.cache %ch0, %ctx[5, 42], %buf
             : !mgp.buffer<<"gpu", 3>>
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    MGP_DeviceBufferType:$buffer,
    UI64Attr:$bBufferSlot
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bBufferSlot `]` `,`
    $buffer attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// BufferFreeOp
//===----------------------------------------------------------------------===//

def BufferFreeOp : MGP_Op<"buffer.free"> {
  let summary = "No-op used to indicate when a buffer is freed.";
  let description = [{
    Has no primitive backing it; it only makes buffer lifetimes explicit, since
    the runtime refcounts buffers.

    Example:
    ```mlir
      %ctx : !mgp.context
      %ch1 : !mef.chain
      %buf0 : !mgp.buffer
      %ch2 = mgp.buffer.free %ch1, %buf0 : !mgp.buffer
    ```
  }];

  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
    `:` custom<BufferTypeToChain>(type($buffer), type($inChain))
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer
  );
  let results = (outs ChainType:$outChain);
}

//===----------------------------------------------------------------------===//
// BufferHostToDeviceOp
//===----------------------------------------------------------------------===//

def BufferHostToDeviceOp : MGP_DeviceOp<"buffer.host_to_device", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Copies a host buffer into a non-host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of an `!mgp.buffer` on the
    host device to an `!mgp.buffer` on a non-host device. The two buffers must
    agree on their runtime size.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed on the device.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %host_buf : !mgp.buffer
      %other_buf : !mgp.buffer<<"gpu", 3>>
      %ch1 = mgp.buffer.host_to_device %ch0, %host_buf, %other_buf
               : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>
      %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$hostBuffer,
    MGP_DeviceBufferType:$otherBuffer,
    MGP_DeviceContextType:$deviceContext,
    MGP_OptDeviceRefAttr:$hostDevice,
    MGP_OptDeviceRefAttr:$otherDevice
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $hostBuffer `,` $otherBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($hostBuffer), type($otherBuffer),
          $hostDevice, $otherDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferHostToDeviceOp::getDevice() { return getOtherDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferMemsetOp
//===----------------------------------------------------------------------===//

def BufferMemsetOp : MGP_DeviceOp<"buffer.memset", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Fills every element of a device buffer with a scalar value.";
  let description = [{
    Sets all `elemSize`-byte elements of `buffer` to the scalar whose
    little-endian bit pattern is the low `elemSize` bytes of `valueBits`. This is
    used to initialize a persistent buffer once during the model's `init` phase.

    The value is carried as a raw bit pattern (rather than a typed attribute) so
    a single primitive can memset any element type: the graph compiler computes
    the pattern from the originating scalar and the runtime replays it with the
    device's memset.

    Example:
    ```mlir
      %ch1 = mgp.buffer.memset(%ch0, %buf, %ctx) elemSize = 4, valueBits = 0
               : (!mef.chain, !mgp.buffer<<"gpu", 3>>, !mgp.device_context<<"gpu", 3>>) -> !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer,
    MGP_DeviceContextType:$deviceContext,
    UI64Attr:$valueBits,
    UI64Attr:$elemSize
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    `(` operands `)` `elemSize` `=` $elemSize `,` `valueBits` `=` $valueBits
        attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    M::DeviceRefAttr BufferMemsetOp::getDevice() {
      return getBuffer().getType().getDevice();
    }
  }];
}

//===----------------------------------------------------------------------===//
// BufferDeviceToHostOp
//===----------------------------------------------------------------------===//

def BufferDeviceToHostOp : MGP_DeviceOp<"buffer.device_to_host", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Copies a non-host buffer into a host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of an `!mgp.buffer` on a
    non-host device to an `!mgp.buffer` on the host device. The two buffers
    must agree on their runtime size.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed. The device must be synchronized with
    `mgp.sync` before the host buffer can be read.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %other_buf : !mgp.buffer<<"gpu", 3>>
      %host_buf : !mgp.buffer
      %ch1 = mgp.buffer.device_to_host %ch0, %other_buf, %host_buf
               : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer
      %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$otherBuffer,
    MGP_DeviceBufferType:$hostBuffer,
    MGP_DeviceContextType:$deviceContext,
    MGP_OptDeviceRefAttr:$otherDevice,
    MGP_OptDeviceRefAttr:$hostDevice
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $otherBuffer `,` $hostBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($otherBuffer), type($hostBuffer),
          $otherDevice, $hostDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferDeviceToHostOp::getDevice() { return getOtherDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferDeviceToDeviceOp
//===----------------------------------------------------------------------===//

def BufferDeviceToDeviceOp : MGP_DeviceOp<"buffer.device_to_device", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Copies a non-host buffer into another non-host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of one `!mgp.buffer` into
    another `!mgp.buffer`. The two buffers must have the same device label,
    but may be on distinct virtual devices.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed on the device.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %src_buf : !mgp.buffer<<"gpu", 3>>
      %dst_buf : !mgp.buffer<<"gpu", 1>>
      %ch1 = mgp.buffer.device_to_device %ch0, %ctx[5, 3], %src_buf, %dst_buf
               : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>
      %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$srcBuffer,
    MGP_DeviceBufferType:$dstBuffer,
    MGP_DeviceContextType:$srcDeviceContext,
    MGP_DeviceContextType:$dstDeviceContext,
    M_DeviceRefAttr:$srcDevice,
    M_DeviceRefAttr:$dstDevice
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $srcBuffer `,` $srcDeviceContext`,` $dstBuffer `,` $dstDeviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($srcBuffer), type($dstBuffer),
          $srcDevice, $dstDevice, type($inChain)) `,` qualified(type($srcDeviceContext))`,` qualified(type($dstDeviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferDeviceToDeviceOp::getDevice() { return getDstDevice(); }
    }];

}


//===----------------------------------------------------------------------===//
// BufferConcatOp
//===----------------------------------------------------------------------===//

def BufferConcatOp : MGP_DeviceOp<"buffer.concat", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Concatenates a sequence of input buffers into an output "
                "buffer.";
  let description = [{
    With a single input this is a buffer copy. Statically known sizes are
    checked for consistency at compile time; inconsistent runtime sizes return
    a runtime error.

    Example:
    ```mlir
    %in_ch : !mef.chain
    %out_buf : !mgp.buffer<<"gpu", 3>>
    %in0_buf : !mgp.buffer<<"gpu", 3>>
    %in1_buf : !mgp.buffer<<"gpu", 3>>
    %in2_buf : !mgp.buffer<<"gpu", 3>>
    %out_ch = mgp.buffer.concat %in_ch, %out_buf, %in0_buf, %in1_buf, %in2_buf
              : (!mgp.buffer<<"gpu", 3>>,
                 !mgp.buffer<<"gpu", 3>>,
                 !mgp.buffer<<"gpu", 3>>) -> !mgp.buffer<<"gpu", 3>>
    %out_ch : !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$outBuffer,
    Variadic<MGP_DeviceBufferType>:$inBuffers,
    MGP_DeviceContextType:$ctx,
    MGP_OptDeviceRefAttr:$bDevice  // 1
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    `(` operands `)` (`device` $bDevice^)? attr-dict `:`
        functional-type(operands, results)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferConcatOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferPlanOp
//===----------------------------------------------------------------------===//
def BufferPlanOp : MGP_Op<"buffer.plan", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Runtime memory planning for buffers";
  let description = [{
    Given the size and sharing information for a list of allocations, returns
    the high watermark size and offsets for each allocation.

    canShare is a flat NxN matrix (row-major) where canShare[i*N+j]=1 iff
    allocations i and j can share the same memory slot (i.e. their lifetimes
    do not overlap). The matrix is symmetric with a zero diagonal.
    N = len(staticSizes) + len(runtimeSizes).
  }];

  let arguments = (ins
    Variadic<Index>:$runtimeSizes,
    UI64ArrayElementsAttr:$alignments,
    UI8ArrayElementsAttr:$canShare,
    OptionalAttr<SI64ArrayElementsAttr>:$staticSizes
  );

  let results = (outs
    Index:$highWatermark,
    Variadic<Index>:$offsets
  );

  let assemblyFormat = [{
    `(` $runtimeSizes `)`
      `<` `alignments` `=` qualified($alignments) `,`
      `canShare` `=` qualified($canShare)
      (`,` `staticSizes` `=` qualified($staticSizes)^)? `>`
      attr-dict `:` functional-type($runtimeSizes, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//

def BufferSliceOp : MGP_Op<"buffer.slice", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Borrows a slice of parent buffer on device with dynamically "
                "known offset and size.";
  let description = [{
    Returns a buffer borrowing the `size`-byte section of the input buffer at
    `offset`. No data is copied.

    The device context must be on the same device as the input buffer; backends
    whose runtime only accepts registered pointers register the slice through it.

    Example:
    ```mlir
    %ch:  !mef.chain
    %ctx : !mgp.device_context<<"cpu", 0>>
    %in_buf : !mgp.buffer
    %offset : index
    %size : index
    %out_buf =
        mgp.buffer.slice %in_buf, %offset, %size, %ctx
          : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
    %out_buf : !mgp.buffer
    ```
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$inBuffer,
    Index:$offset,
    Index:$size,
    MGP_DeviceContextType:$deviceContext
  );

  let results = (outs MGP_DeviceBufferType:$outBuffer);

  let assemblyFormat = [{
    $inBuffer `,` $offset `,` $size `,` $deviceContext attr-dict `:`
      type($inBuffer) `,` qualified(type($deviceContext))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferBulkSliceOp
//===----------------------------------------------------------------------===//

def BufferBulkSliceOp : MGP_Op<"buffer.bulk_slice", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let summary = "Bulk slice of a buffer pool into N sub-buffers.";
  let description = [{
    Given a pre-allocated pool buffer and N (offset, size) pairs, produces N
    sub-buffers each borrowing a non-overlapping slice of the pool.
    Semantically equivalent to N individual `mgp.buffer.slice` ops, but
    represented as a single op to avoid stressing sequence fusion with a large
    fan-out of duplicatable slice ops.

    The device context must be on the same device as the base buffer; backends
    whose runtime only accepts registered pointers (e.g. Trainium) register each
    slice through it.

    Example:
    ```mlir
    %base : !mgp.buffer<<"gpu", 0>>
    %ctx : !mgp.device_context<<"gpu", 0>>
    %off0, %off1 : index
    %sz0, %sz1 : index
    %slice0, %slice1 =
        mgp.buffer.bulk_slice %base, %ctx offsets(%off0, %off1) sizes(%sz0, %sz1)
          : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>
    ```
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$baseBuffer,
    MGP_DeviceContextType:$deviceContext,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes
  );

  let results = (outs Variadic<MGP_DeviceBufferType>:$outBuffers);

  let assemblyFormat = [{
    $baseBuffer `,` $deviceContext `offsets` `(` $offsets `)`
      `sizes` `(` $sizes `)` attr-dict
      `:` type($baseBuffer) `,` qualified(type($deviceContext))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferToIndexOp
//===----------------------------------------------------------------------===//

def BufferToIndexOp : MGP_Op<"buffer.to_index", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  Pure,
]> {
  let summary = "Returns contents of the buffer to an `index`";
  let description = [{
    Returns contents of a 4- or 8-byte buffer as a 64-bit `index`.

    Sign extension is performed when upcasting from a 32-bit integer.

    Example:
    ```mlir
    %0 = mgp.buffer.to_index %ch0, %buf : !mgp.buffer
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$inBuffer
  );

  let results = (outs Index:$out);

  let assemblyFormat = [{
     $inChain `,` $inBuffer attr-dict `:` type($inBuffer)
  }];
}

//===----------------------------------------------------------------------===//
// BufferToBoolOp
//===----------------------------------------------------------------------===//

def BufferToBoolOp : MGP_DeviceOp<"buffer.to_bool", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Returns contents of a 1-byte buffer as an i1.";
  let description = [{
    Example:
    ```mlir
    %0 = mgp.to_bool %ch0, %ctx[5], %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer,
    MGP_OptDeviceRefAttr:$device
  );
  let results = (outs I1:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
      `:` custom<BufferTypeToDevice>(type($buffer), $device, type($inChain))
  }];
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//

def BufferSetWithIndexOp : MGP_DeviceOp<"buffer.set_with_index", [
  // TODO(#37615): Declare MGP_LiftableChainCreateOpInterface.
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Set a buffer with the values from index inputs.";
  let description = [{
    The runtime index values will be sign extended or truncated based on the
    given buffer's size. Therefore only buffer sizes of 4 and 8 times the number
    of index inputs are supported by the primitive.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf0 : !mgp.buffer<<"gpu", 3>>
    %idx0 : index
    %ch0 = mgp.buffer.set_with_index %buf0, %idx0 : !mgp.buffer<<"gpu", 3>>
    %ch0 : !mef.chain

    %idx1 : index
    %idx2 : index
    %buf1 : !mgp.buffer<<"gpu", 3>>
    %ch1 = mgp.buffer.set_with_index %buf1, %idx0, %idx1, %idx2 : !mgp.buffer<<"gpu", 3>>
    %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$buffer,
    Variadic<Index>:$values,
    MGP_OptDeviceRefAttr:$device
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{$buffer ( `,` $values^ )? attr-dict
    `:` custom<BufferTypeToDevice>(type($buffer), $device)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

def DebugTensorPrintOp : MGP_Op<"debug.tensor.print", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
    "getLiftableAttributes"
  ]>,
]> {
  let summary = "Prints a debug representation of an tensor. For debugging "
                "and testing only.";
  let description = [{
    The tensor is given as an `!mgp.buffer` and `!mgp.tensor_spec`. If a label
    is supplied the contents are printed with that label, otherwise just the
    metadata. Only CPU buffers can be printed with contents; others may need an
    explicit device-to-host copy.

    Note this op accepts and returns `!mef.chain`s since printing is managed
    by the host.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %buf: !mgp.buffer<<"gpu", 3>>
    %spec: !mgp.tensor_spec<8x64xf32>
    %ch1 = mgp.debug.tensor.print %ch0, %ctx[5], %buf, %spec, "label"
           : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %ch1: !mef.chain
    ```
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_DeviceBufferType:$buffer,
    MGP_TensorSpecType:$spec,
    DefaultValuedAttr<StrAttr, "\"\"">:$label
  );
  let results = (outs ChainType:$outChain);

  let extraClassDefinition = [{
    SmallVector<Attribute> DebugTensorPrintOp::getLiftableAttributes() {
      return {getLabelAttr()};
    }
  }];

  let assemblyFormat = [{
    $inChain `,` $buffer `,` $spec
    ( `,` $label^ )? attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// SyncOp
//===----------------------------------------------------------------------===//

def MGP_SyncOp : MGP_DeviceOp<"sync", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
  DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getIgnorableAttributes"]>,
  Pure,
]> {
  let summary = "Schedules a device specific sync";
  let description = [{
    Returns a ready chain once the device has synchronized. In
    cpu this is a nop as the sync happens with the chain, on
    gpu this records a gpu event.
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    MGP_DeviceContextType:$deviceContext,
    MGP_OptDeviceRefAttr:$bDevice  // 1
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    `(` operands `)` (`device` $bDevice^)? attr-dict `:`
        functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    SmallVector<mlir::Attribute> SyncOp::getIgnorableAttributes() {
      return {getBDeviceAttr()};
    }

    M::DeviceRefAttr SyncOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// DeviceWaitOp
//===----------------------------------------------------------------------===//

def MGP_DeviceWaitOp : MGP_DeviceOp<"device_wait", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  Pure,
]> {
  let summary = "Enqueues a one-directional cross-stream wait";
  let description = [{
    Makes the waiting device context's stream wait for the work already
    enqueued on the signaling device context's stream. It records an event on
    the signaling stream and enqueues a wait for that event on the waiting
    stream.

    Unlike `mgp.sync`, this does not block the host; it only enqueues a
    cross-stream dependency. It is used to make the synchronization required by
    cross-device buffer transfers explicit in the graph instead of hidden
    inside the driver's device-to-device copy implementation.

    The `bDevice` attribute is the waiting device (the device whose stream
    receives the wait). It must be an accelerator for the op to be meaningful
    and is used for the op's device dispatch.

    The `isSideStreamJoin` attribute marks the wait that joins a side stream
    back into its base stream. MOGGToMGP moves such a wait to the first
    consumer of the side stream's results, and LowerPrimitives keeps it out of
    sequence fusion.
  }];

  let arguments = (ins
    ChainType:$inChain,
    MGP_ContextType:$theContext,
    MGP_DeviceContextType:$waitingDeviceContext,
    MGP_DeviceContextType:$signalingDeviceContext,
    M_DeviceRefAttr:$bDevice,
    UnitAttr:$isSideStreamJoin
  );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    `(` operands `)` `device` $bDevice (`isSideStreamJoin` $isSideStreamJoin^)?
        attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    M::DeviceRefAttr DeviceWaitOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// DeviceContextSelectStreamOp
//===----------------------------------------------------------------------===//

def MGP_DeviceContextSelectStreamOp : MGP_Op<"device_context.select_stream", [
  AllTypesMatch<["deviceContext", "view"]>,
  Pure,
]> {
  let summary = "Returns a device-context view bound to a given stream.";
  let description = [{
    Returns a "retain-all" view of the input device context whose current
    stream is `streamId`. The view shares the input context's full stream set,
    driver context, and device memory pool; only the current-stream selector
    differs.

    It binds a stream-tagged subgraph call to a side stream without threading
    per-stream device contexts through the model: the base device context is
    already an implicit model argument, so the view is derived at the call
    site. Nested selections compose since the view keeps the full stream set.

    Backends without a multi-stream model return a view equivalent to the input
    context, so all work runs on its single stream.

    It is executed directly by the MGP runtime (a primitive registered in
    `registerMGPPrimitives`, like `mgp.runtime.create`): it implements no
    `EmitsMojoInterface`, so `mgp-lower-primitives` leaves it intact and the
    executor dispatches it by name to its C++ primitive.
  }];

  let arguments = (ins
    MGP_DeviceContextType:$deviceContext,
    UI64Attr:$streamId
  );
  let results = (outs MGP_DeviceContextType:$view);
  let assemblyFormat = [{
    `[` $streamId `]` $deviceContext attr-dict `:`
        qualified(type($deviceContext))
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
