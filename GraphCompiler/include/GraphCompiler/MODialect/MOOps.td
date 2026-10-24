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
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD

include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "GraphCompiler/MODialect/MOEnums.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MOSHDialect/MOSHTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []> :
  Op<MODialect, mnemonic, [MO_BufferOpConformance] # traits>;

// A compile-time f32 scalar, stored as a rank-0 elements attribute (e.g.
// `#M.dense_array`) rather than a builtin `FloatAttr`: `max._core` static-links
// its own LLVM, and its bytecode reader and writer reject `APFloat` semantics
// owned by libmax (GEX-4052).
def MO_F32ScalarAttr : ConfinedAttr<ElementsAttr, [AttrConstraint<
  CPred<"::llvm::cast<::mlir::ElementsAttr>($_self).getShapedType()"
        ".getRank() == 0 && ::llvm::cast<::mlir::ElementsAttr>($_self)"
        ".getElementType().isF32()">,
  "rank-0 f32 elements attribute">]>;

// Base class for ops that work with tensors in the MO dialect.
class MO_TensorOp<string mnemonic, list<Trait> traits = []> :
  MO_Op<mnemonic, [
    MO_HasDeviceInterface,
    MO_TensorOpInterface,
  ] # traits>;

// Base class for ops that work with tensors but do not have a notion of device.
class MO_DeviceAgnosticTensorOp<string mnemonic, list<Trait> traits = []> :
  MO_Op<mnemonic, [MO_TensorOpInterface] # traits>;

// Base class for ops working with control flow
class MO_ControlOp<string mnemonic, list<Trait> traits = []> :
  MO_TensorOp<mnemonic, [
    MO_ControlOpInterface,
    MO_ExecutesOnHost,
    MO_PreservedDuringKernelLowering,
  ] # traits>;

// Base class for ops working with mutable buffers
class MO_MutableOp<string mnemonic, list<Trait> traits = []> :
  MO_Op<mnemonic, [
    DeclareOpInterfaceMethods<MO_MutableOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    MO_HasDeviceInterface,
  ] # traits>;

// Base class for ops working with mutable buffers that do not have a notion of device
class MO_DeviceAgnosticMutableOp<string mnemonic, list<Trait> traits = []> :
  MO_Op<mnemonic, [
    DeclareOpInterfaceMethods<MO_MutableOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  ] # traits>;

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

def MO_IndexToTensorOp : MO_Op<"index.to_tensor", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "`scalar<si64>` to scalar `!mo.tensor` casting operation";
  let description = [{
    Example:

    ```mlir
      %c: scalar<si64>
      %scalarT = mo.index.to_tensor(%c) -> !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins SI64Scalar:$input);
  let results = (outs MO_RankedScalarIndexTensor:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

def MO_TransferOp : MO_DeviceAgnosticTensorOp<"transfer", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Device transfer operation";
  let description = [{
    Copies or aliases the operand so it is available on the result's (virtual)
    device. Source and destination may be the same device: with
    `alwaysElideSameDeviceCopy` set this never copies, otherwise copy vs. alias
    is implementation defined.

    Example:

    ```mlir
      %arg : !mo.tensor<[N, 8], f32, gpu:3>
      %onOtherDevice = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:3> to <"gpu", 1>
      %onHost = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:0> to <"cpu", 1>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$alwaysElideSameDeviceCopy,
    MO_Chain:$inChain
  );
  let results = (outs MO_Tensor:$result, MO_Chain:$outChain);

  let assemblyFormat = [{
    `[` $inChain `]` $input `:` qualified(type($input)) `to`
    custom<TransferResult>(ref(type($input)), type($result))
    (` ` `[` `alwaysElideSameDeviceCopy` `=` $alwaysElideSameDeviceCopy^ `]`)?
    attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins
      "TensorValue":$input,
      "DeviceRefAttr":$destDevice,
      "ChainValue":$inChain,
      CArg<"bool", "true">:$alwaysElideSameDeviceCopy
    )>,
  ];
}

//===----------------------------------------------------------------------===//
// StableCreateOp
//===----------------------------------------------------------------------===//

def MO_StableCreateOp : MO_DeviceAgnosticTensorOp<"stable.create", [
  MO_PreservedDuringKernelLowering,
  Pure,
  TypesMatchWith<"input is the result without the stable marker",
                 "result", "input",
                 "::llvm::cast<TensorType>($_self).cloneWithStable(false)">,
]> {
  let summary = "Marks a tensor as read through a stable address slot.";
  let description = [{
    This op annotates a tensor value as a stable tensor. Stable tensors are
    backed by storage slots associated with a device graph region. This
    operation encodes the assignment of a tensor to a stable storage slot
    before lowering to MOGG/MGP obscures the kernel signature.

    Example:

    ```mlir
      %arg : !mo.tensor<[N, 8], f32, gpu:1>
      %stable = mo.stable.create %arg : !mo.tensor<[N, 8], f32, gpu:1, {stable}>
    ```
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";

  let builders = [
    OpBuilder<(ins "TensorValue":$input), [{
      build($_builder, $_state, input.getType().cloneWithStable(), input);
    }]>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_DeviceAgnosticTensorOp<"graph", [
  // needed to leverage KGEN parameter infra
  DeclInterface,
  HasOnlyGraphRegion,
  HasParent<"ModuleOp">,
  IsolatedFromAbove,
  MO_PreservedDuringKernelLowering,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"OutputOp">,
  SymbolName,
  SymbolVisibility,
  Symbol,
]> {
  let summary = "Computation graph operation";
  let description = [{
    A computation graph: typed inputs and outputs, a body of ops, and a
    terminating `mo.output`.

    Example:

    ```mlir
      mo.graph @example<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        // ... intermediate computations ...
        %res : !mo.tensor<[D3], f32>
        mo.output<D3> %arg : !mo.tensor<[D3], f32>
      }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParameters,
    KGEN_ParamDeclArrayAttr:$resultParameters,
    IntProp<"int64_t">:$counter,
    UnitAttr:$isSubgraph,
    UnitAttr:$isDeviceGraph
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<GraphOp>($sym_visibility, $sym_name, $signature, $functionType,
                    $inputParameters, $resultParameters, $isSubgraph,
                    $isDeviceGraph, attr-dict, $body, $counter)
  }];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
    OutputOp getOutputOp();

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// Returns a new `IndexType` parameter named `prefix...`, unique among
    /// this graph's parameters.
    KGEN::ParamDeclAttr getNewDimParam(const Twine &prefix);

    /// Like `getNewDimParam`, but of the given type.
    KGEN::ParamDeclAttr getNewParamOfType(const Twine &prefix, Type type);

    std::string getNewParamName(const Twine &prefix);

    /// Returns the #M.device_spec_collection annotation, or null.
    DeviceSpecCollectionAttr getDeviceSpecs();
    void setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs);
    void setDeviceTargetInfo(TargetInfoAttr targetInfo);
    TargetInfoAttr getDeviceTargetInfo();

    /// DeclInterface methods, backed by MO's own parameter implementation.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "TypeRange":$inputTypes,
      "TypeRange":$resultTypes,
      CArg<"bool", "false">:$isSubgraph
    )>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def MO_CallOp : MO_ControlOp<"call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<MO_ConditionallyInPlaceInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, [
    "getInChains",
    "getInChainsMutable",
    "getOutChains",
  ]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasParent<"MO::GraphOp">,
  MO_ExecutesOnHost,
  MO_ParamDeclaration,
]> {
  let summary = "Call a computation graph";
  let description = [{
    Example:

    ```mlir
      %res = mo.call @gelu(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    FlatSymbolRefAttr:$callee,
    DefaultValuedAttr<StrAttr, "\"\"">:$prefix,
    MO_ParamDecls:$outputParamDecls,

    // Needed to satisfy CallOpInterface
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) $callee `(` operands `)`
       `` custom<CallAttrs>($prefix, $arg_attrs, $res_attrs) attr-dict `:`
       functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      this->setCalleeAttr(cast<FlatSymbolRefAttr>(cast<SymbolRefAttr>(callee)));
    }
  }];

  let hasVerifier = true;
}


//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MO_OutputOp : MO_ControlOp<"output", [
  DeclareOpInterfaceMethods<ParamOpInterface>,
  HasParent<"MO::GraphOp">,
  ReturnLike,
  Terminator,
]>, NoParamDecls {
  let summary = "Graph output operation";
  let description = [{
    Terminates a `mo.graph` with its output parameters and values. Operands
    must match the enclosing graph's signature after substituting the output
    parameter bindings.

    Examples:

    ```mlir
      mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
        mo.output %arg0 : !mo.tensor<?, f32>
      }

      mo.graph @with_params<D1 -> D2>(
          %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
      }
    ```

    Operand types and the graph's output types need not match textually; the
    compiler verifies they match after substitution.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
  );

  let assemblyFormat = [{
    `` custom<OutputValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : MO_ControlOp<"if", [
  DeclInterface,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getEntrySuccessorRegions",
    "getSuccessorInputs",
  ]>,
  HasOnlyGraphRegion,
  NoRegionArguments,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"MO::YieldOp">,
]> {
  let summary = "`if` control flow operation";
  let description = [{
    Runs the 'then' block if the `i1` condition is true, else the 'else'
    block, and returns that block's `mo.yield` values, which must match the
    result types. Blocks may use outer values.

    Example:

    ```mlir
      %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
        %v1 = mo.add(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                ) -> !mo.tensor<?, f32>
        mo.yield %v1 : !mo.tensor<?, f32>
      } else {
        %v2 = mo.sub(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                  ) -> !mo.tensor<?, f32>
        mo.yield %v2 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[MO_CondLike, MO_BoolScalar]>:$cond,
    MO_ParamDecls:$outputParamDecls
  );
  // Forwarded 1:1 from `mo.yield`, which excludes `!mo.buffer` (no alias
  // analysis for forwarded buffers).
  let results = (outs Variadic<MO_AnyNonBufferType>:$results);
  let regions = (region
    SizedRegion<1>:$thenRegion,
    SizedRegion<1>:$elseRegion
  );

  let extraClassDeclaration = [{
    Block *getThenBlock();
    Block *getElseBlock();

    /// DeclInterface methods, backed by MO's own parameter implementation.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `(` $cond `:` qualified(type($cond)) `)` (`->` type($results)^)?
    $thenRegion `else` $elseRegion attr-dict
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : MO_ControlOp<"while", [
  DeclInterface,
  DeclareOpInterfaceMethods<LoopLikeOpInterface, [
    "getInitsMutable",
    "getRegionIterArgs",
    "getYieldedValuesMutable",
  ]>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, [
    "getInChains",
    "getInChainsMutable",
    "getOutChains",
  ]>,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getEntrySuccessorOperands",
    "getSuccessorInputs",
  ]>,
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlock,
]> {
  let summary = "`while` control flow operation";
  let description = [{
    Runs the "body" block while the "cond" block's condition is true. Both
    blocks may use the signature values and outer values.

    - "cond" terminates with `mo.while.condition`: a `!mo.tensor<[], bool>`
      followed by values matching the signature.
    - "body" yields values matching the signature. Every input needs an "as"
      clause renaming it to a fresh symbol.

    The results are the `mo.while.condition` operands when the condition is
    false.

    Example:

    ```mlir
      %x: !mo.tensor<[2], f32>
      %y: !mo.tensor<[], f32>
      %res = mo.while (%x as %inner_x: !mo.tensor<[2], f32>) {
        %zero = mo.constant <#M.dense_array<0> : tensor<1xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
        %shape = mo.constant <#M.dense_array<> : tensor<0xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[0], si64>
        %mean0 = mo.mean (%inner_x, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
        %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
        %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
        mo.while.condition(%cond) %inner_x : !mo.tensor<[2], f32>
      } do {
        %new_x = mo.add(%inner_x, %inner_x) : !mo.tensor<[2], f32>
        mo.yield %new_x : !mo.tensor<[2], f32>
      }
      %res: !mo.tensor<[2], f32>
    ```
  }];

  // Shares one signature with `mo.yield` and `mo.while.condition`, which
  // exclude `!mo.buffer`.
  let arguments = (ins Variadic<MO_AnyNonBufferType>:$inputs);
  let results = (outs Variadic<MO_AnyNonBufferType>:$results);
  let regions = (region
    SizedRegion<1>:$condRegion,
    SizedRegion<1>:$bodyRegion
  );

  let extraClassDeclaration = [{
    WhileConditionOp getCondTerminator();
    YieldOp getBodyTerminator();

    Block *getCondBlock();
    Block *getBodyBlock();

    Block::BlockArgListType getCondArguments();
    Block::BlockArgListType getBodyArguments();

    /// DeclInterface methods, backed by MO's own parameter implementation.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  // Custom because inputs are both referential and binding occurrences.
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def MO_YieldOp : MO_ControlOp<"yield", [
  DeclareOpInterfaceMethods<ParamOpInterface>,
  ParentOneOf<["IfOp", "WhileOp", "ParallelOp", "SequenceOp"]>,
  Pure,
  ReturnLike,
  Terminator,
]>, NoParamDecls {
  let hasVerifier = 1;
  let summary = "`yield` operation";
  let description = [{
    Returns values from a control flow block.

    Example:

    ```mlir
      mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
        mo.yield %arg0 : !mo.tensor<?, f32>
      } else {
        mo.yield %arg1 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins
    Variadic<MO_AnyNonBufferType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
  );

  let assemblyFormat = [{
    `` custom<OutputValues>($parameters) attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

def MO_WhileConditionOp : MO_ControlOp<"while.condition", [
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
    "getSuccessorRegions",
  ]>,
  HasParent<"WhileOp">,
  MO_PreservedDuringKernelLowering,
  Pure,
  Terminator,
]> {
  let summary = "loop continuation condition";
  let description = [{
    Terminates the `mo.while` cond block with the continuation condition and
    values matching the parent's operands. See `mo.while` for an example.
  }];

  let arguments = (ins
    MO_CondLike:$condition,
    Variadic<MO_AnyNonBufferType>:$args
  );

  let assemblyFormat = [{
    `(` $condition `)` attr-dict ($args^ `:` type($args))?
  }];
}

//===----------------------------------------------------------------------===//
// Bundle / Unbundle ops
//===----------------------------------------------------------------------===//

// Bundle/unbundle ops use Op<> directly (not MO_Op) to avoid
// MO_BufferOpConformance — these are pure grouping ops with no mutation.

def MO_TensorBundleOp : Op<MODialect, "tensor.bundle", [
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Bundle individual tensors into an mo.bundle.";
  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs MO_Bundle:$result);
  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs)>,
  ];
}

def MO_TensorUnbundleOp : Op<MODialect, "tensor.unbundle", [
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Extract individual tensors from an mo.bundle.";
  let arguments = (ins MO_Bundle:$input);
  let results = (outs Variadic<MO_Tensor>:$outputs);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, $outputs)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

def MO_ParallelOp : MO_TensorOp<"parallel", [
  AttrSizedOperandSegments,
  DeclInterface,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, [
    "getInChains",
    "getInChainsMutable",
    "getOutChains",
  ]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  HasOnlyGraphRegion,
  MO_ExecutesOnHost, // TODO this should be MO_MultiDevice
  MO_PreservedDuringKernelLowering,
  RegionKindInterface,
  SingleBlock,
  SingleBlockImplicitTerminator<"MO::YieldOp">,
]> {
  let summary = "`parallel` region operation";
  let description = [{
    Runs the body once per launch. Each input is an `!mo.bundle` of per-device
    values; all bundles have one element per launch. The body gets one block
    argument per bundle, typed as the bundle's first element. Each yield
    operand becomes one `!mo.bundle` result with per-launch devices from
    `launchDevices`.

    `launchDevices` is the source of truth for per-launch devices, printed as
    `devices(...)`. The clause may be elided when it matches `inputs[0]`, and
    is required when there are no inputs. Every input and result bundle must
    agree with it element-by-element.

    The optional `buffers(...)` clause holds one signal buffer per launch for
    collectives (e.g. allreduce). Buffers are operands for memory-effect
    tracking only; the body captures them from the enclosing scope rather
    than as block arguments.

    `buffers(...)` and `chain(...)` are both present or both absent. When
    present, the chain is threaded explicitly through the body: a trailing
    `!mo.chain` block argument carries the in-chain, side-effecting ops thread
    it, and `mo.yield`'s last operand becomes the op's trailing `!mo.chain`
    result (completion of all launches).

    Example with one bundle input (no buffers, no chain):
    ```mlir
    %dt = mo.tensor.bundle(%a, %b) : (...) -> (...)
    // devices(...) elided; auto-derived from %dt.
    %res = mo.parallel (%arg) in (%dt : !mo.bundle<[...]>)
        -> (!mo.bundle<[...]>) {
      %1 = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```

    Example with buffers and chain (bundled allreduce):
    ```mlir
    %dt = mo.tensor.bundle(%a, %b) : (...) -> (...)
    %res, %ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[...]>)
        buffers(%s0 : !mo.buffer<[1], ui8, gpu:0>,
                %s1 : !mo.buffer<[1], ui8, gpu:1>)
        chain(%ch_in)
        -> (!mo.bundle<[...]>) {
      %p0, %p1 = mo.bundled.expand(%arg) : ...
      %out, %ch1 = mo.bundled.allreduce.sum(%p0, %p1, %s0, %s1, %bch) : ...
      mo.yield %out, %ch1 : !mo.tensor<[3], f32, gpu:0>, !mo.chain
    }
    ```

    Example with no bundle inputs (`devices(...)` required):
    ```mlir
    %res = mo.parallel () in () devices(gpu:0-1)
        -> (!mo.bundle<[...]>) {
      %1 = mo.relu(%a) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```
  }];

  let arguments = (ins
    Variadic<MO_Bundle>:$inputs,
    Variadic<MO_Buffer>:$buffers,
    Optional<MO_Chain>:$inChain,
    M_DeviceRefArrayAttr:$launchDevices
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Bundle, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    /// Builds an empty body with one block argument per bundle input.
    /// `launchDevices` is derived from `inputs.front()`.
    OpBuilder<(ins
      "ValueRange":$inputs,
      "TypeRange":$resultTypes
    )>,
    /// As above; `buffers` and `inChain` are both present or both absent.
    OpBuilder<(ins
      "ValueRange":$inputs,
      "ValueRange":$buffers,
      "Value":$inChain,
      "TypeRange":$resultTypes
    )>,
    /// For callers that know the device list directly (e.g. non-bundle
    /// sources or empty `inputs`).
    OpBuilder<(ins
      "ValueRange":$inputs,
      "ValueRange":$buffers,
      "Value":$inChain,
      "TypeRange":$resultTypes,
      "ArrayRef<DeviceRefAttr>":$launchDevices
    )>,
  ];

  let extraClassDeclaration = [{
    /// DeclInterface methods, backed by MO's own parameter implementation.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();

    SmallVector<DeviceRefAttr> getLaunchDeviceList() {
      return llvm::to_vector(getLaunchDevices().getAsRange<DeviceRefAttr>());
    }

    size_t getNumLaunches() {
      return getLaunchDevices().size();
    }

    bool hasBuffers() { return !getBuffers().empty(); }

    /// True iff `hasBuffers()`.
    bool hasChain() { return static_cast<bool>(getInChain()); }

    ChainValue getOutChain() {
      assert(hasChain() && "no chain on this parallel op");
      return cast<ChainValue>(getResults().back());
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

def MO_SequenceOp : MO_TensorOp<"sequence", [
  DeclInterface,
  HasOnlyGraphRegion,
  MO_ExecutesOnHost,
  MO_PreservedDuringKernelLowering,
  RegionKindInterface,
  SingleBlock,
  SingleBlockImplicitTerminator<"MO::YieldOp">,
]> {
  let summary = "`sequence` region bound to a device stream";
  let description = [{
    Binds a region of ops to the device stream `streamId` (0 is the default).
    The region is a fusion boundary, and MOGG->MGP lowering binds the body's
    device context to a `mgp.device_context.select_stream` view.

    Operands map 1:1 to block arguments and `mo.yield` maps 1:1 to results.
    The body is an unordered graph region; chains encode mutable ordering.

    Example:
    ```mlir
    %out = mo.sequence[1] (%a : !mo.tensor<[3], f32, gpu:0>) (%arg)
        -> (!mo.tensor<[3], f32, gpu:0>) {
      %1 = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
      mo.yield %1 : !mo.tensor<[3], f32, gpu:0>
    }
    ```
  }];

  // No buffers cross the boundary; chains carry mutable ordering.
  let arguments = (ins
    Variadic<MO_AnyNonBufferType>:$inputs,
    DefaultValuedAttr<UI64Attr, "0">:$streamId
  );
  let results = (outs Variadic<MO_AnyNonBufferType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    /// Builds an empty body with one argument per input; callers add the
    /// `mo.yield`.
    OpBuilder<(ins
      "ValueRange":$inputs,
      "TypeRange":$resultTypes,
      "uint64_t":$streamId
    )>,
  ];

  let extraClassDeclaration = [{
    /// DeclInterface: this op declares no parameters of its own.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  let assemblyFormat = [{
    `[` $streamId `]` `(` $inputs `:` type($inputs) `)` attr-dict
        custom<SequenceRegion>($bodyRegion, ref(type($inputs)), type($results))
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def MO_RebindOp : MO_ControlOp<"rebind", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Parameter rebind operation";
  let description = [{
    Rebinds a `!mo.tensor` type (typically its shape) to another so strict
    shape verifiers type-check. Unlike a cast, no data conversion happens; the
    types are assumed equal at runtime. Where that is not statically provable,
    runtime assertions must be inserted separately; this op does not add them.

    Rebinds that declare parameters "rename" dimensions. Shape inference
    removes them unless they name unknown (`?`) dims from third-party
    dialects like PT.

    Examples:
    ```mlir
      // something like ASSERT N == 3 AND Sh == {3, 1} if we aren't statically
      // sure of this fact.
      %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
      %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[3, 1], f32>

      // Renames dims. K == 3, M == 1
      %3 = mo.rebind<() -> K, M>(%0) :
        !mo.tensor<[3, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantExternalOp
//===----------------------------------------------------------------------===//

def MO_ConstantExternalOp : MO_TensorOp<"constant.external", [
  MO_ConstantLike,
  MO_ExplicitDevice,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "External constant operation";
  let description = [{
    References an external constant whose data is unknown at compile time but
    whose other tensor attributes are static. Names are unique per `mo.graph`.

    `device` is where the weights registry pointers reside. It may differ from
    the result type's device only when it is the host, to support mmap'ed
    weights.

    TODO(MSDK-1060): check the device at runtime by storing it with the
    registry pointer, and emit an explicit mo.transfer in the graph API
    instead of an implicit HtoD copy when lowering to
    mgp.buffer.constant.external.

    With `hasAlias`, the constant may be updated through an external alias
    between invocations, so it is not lifted to the init phase.

    Example:

    ```mlir
    %weight = mo.constant.external
      [name = "foo", align = 16, device = #M.device_ref<"gpu", 0>]
      : !mo.tensor<[4096, 4096], bf16>
  }];

  let arguments = (ins
    StrAttr:$name,
    UI64Attr:$align,
    M_DeviceRefAttr:$device,
    DefaultValuedAttr<BoolAttr, "false">:$hasAlias,
    DefaultValuedAttr<BoolAttr, "false">:$isPlaceholder
  );
  let results = (outs MO_Tensor:$result);

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    `` custom<ConstantExternalAttrs>($name, $align, $device, $hasAlias,
                                     $isPlaceholder)
    attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_TensorOp<"constant", [
  ConstantLike,
  MO_ConstantLike,
  MO_PreservedDuringKernelLowering,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Constant operation";
  let description = [{
    A literal value inside the graph.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
    ```
  }];

  let arguments = (ins
    ConfinedAttr<ElementsAttr, [AnyAttrOf<[
      Builtin_DenseResourceElementsAttr,
      M_AlignedBytesAttr,
      M_ArrayElementsAttr,
    ]>]>:$value
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "` ` `<` $value `>` attr-dict `:` qualified(type($result))";

  let hasVerifier = 1;
  let hasFolder = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    /// Takes the device attribute from the result type, if any.
    OpBuilder<(ins "Type":$resultType, "ElementsAttr":$value)>,

    // Array overloads build rank-1 tensors; scalar overloads build rank-0.
    OpBuilder<(ins
      "ArrayRef<int32_t>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "int32_t":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "ArrayRef<int64_t>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "int64_t":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "ArrayRef<APFloat>":$values,
      "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "APFloat":$value,
      "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "ArrayRef<APInt>":$values,
      "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "APInt":$value,
      "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "float":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "double":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "ArrayRef<bool>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
    OpBuilder<(ins
      "bool":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef
    )>,
  ];
}

def MO_ConstantScalarOp : MO_TensorOp<"constant.scalar", [
  ConstantLike,
  MO_HostOnly,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Scalar constant operation";
  let description = [{
    Same as `mo.constant`, but specialized to scalar types.

    Example:

    ```mlir
      %0 = mo.constant.scalar <3 : si64> : !mo.scalar<si64>
    ```
  }];

  let arguments = (ins
    AnyAttrOf<[
      BoolAttr,
      Builtin_FloatAttr,
      Builtin_IntegerAttr,
    ]>:$value
  );
  let results = (outs MO_Scalar:$result);

  let assemblyFormat = "` ` `<` $value `>` attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
  let hasVerifier = true;

  let builders = [
    OpBuilder<(ins "TypedAttr":$value)>,
  ];
}

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

// For ops where each output depends only on the corresponding indices in the inputs
class MO_ElementWiseOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    MO_ElementWiseLike,
    MO_ParamDeclaration,
    MO_SameOperandsAndResultShape,
    MO_SingleDevice,
    Pure,
  ]> {
  let results = (outs MO_Tensor:$result);
}

// Like MO_ElementWiseOp, but the dtypes and shapes match for all results/operands
class MO_ElementWiseMonoTypeOp<string name, list<Trait> traits = []> :
  MO_ElementWiseOp<name, traits # [MO_SameOperandsAndResultType]> {
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

class MO_ElementWiseUnaryOp<string name, list<Trait> traits = []> :
  MO_ElementWiseOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseUnaryMonoTypeOp<string name, list<Trait> traits = []> :
  MO_ElementWiseMonoTypeOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseUnaryFloatMonoTypeOp<string name, list<Trait> traits = []> :
  MO_ElementWiseUnaryMonoTypeOp<name, traits> {
  let arguments = (ins MO_FloatTensor:$input);
}

class MO_ElementWiseBinaryOp<string name, list<Trait> traits = []> :
  MO_ElementWiseOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ElementWiseBinaryMonoTypeOp<string name, list<Trait> traits = []> :
  MO_ElementWiseMonoTypeOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ComparisonOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    MO_ParamDeclaration,
    MO_SameOperandsAndResultShape,
    MO_SameOperandsElementType,
    MO_SingleDevice,
    Pure,
  ]> {
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class MO_ComparisonUnaryOp<string name, list<Trait> traits = []> :
  MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x);
}

class MO_ComparisonBinaryOp<string name, list<Trait> traits = []> :
  MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_LogicalBinaryOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    Commutative,
    MO_ElementWiseBinary,
    MO_ParamDeclaration,
    MO_SameOperandsAndResultType,
    MO_SingleDevice,
    Pure,
  ]> {
  let arguments = (ins MO_BooleanMask:$input_x, MO_BooleanMask:$input_y);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_AddOp : MO_ElementWiseBinaryMonoTypeOp<"add", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Add operation";
  let description = [{
    Returns `x + y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_AbsOp : MO_ElementWiseUnaryMonoTypeOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_DivOp : MO_ElementWiseBinaryMonoTypeOp<"div", [
  MO_ExpensiveElementwise,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Divide operation";
  let description = [{
    Returns `x / y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ExpOp : MO_ElementWiseUnaryMonoTypeOp<"exp", [MO_ExpensiveElementwise]> {
  let summary = "Exponential function operation";
  let description = [{
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ErfOp : MO_ElementWiseUnaryMonoTypeOp<"erf", [MO_ExpensiveElementwise]> {
  let summary = "(Gaussian) error function operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}


def MO_LogOp : MO_ElementWiseUnaryMonoTypeOp<"log", [MO_ExpensiveElementwise]> {
  let summary = "Natural logarithm operation";
  let description = [{
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_Log1pOp : MO_ElementWiseUnaryMonoTypeOp<"log1p", [MO_ExpensiveElementwise]> {
  let summary = "Natural logarithm of `1 + x` operation ";
  let description = [{
    Returns `log(1 + x)`, accurate for small `x` where `1 + x` rounds to 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MaxOp : MO_ElementWiseBinaryMonoTypeOp<"max", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Maximum operation";
  let description = [{
    Returns `max(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MinOp : MO_ElementWiseBinaryMonoTypeOp<"min", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Minimum operation";
  let description = [{
    Returns `min(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ModOp : MO_ElementWiseBinaryMonoTypeOp<"mod", [MO_ExpensiveElementwise]> {
  let summary = "Modulus operation";
  let description = [{
    Returns `x mod y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], si32>
      %rhs: !mo.tensor<[2, 3], si32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], si32>
    ```
  }];
}

def MO_MulOp : MO_ElementWiseBinaryMonoTypeOp<"mul", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Multiplication operation";
  let description = [{
    Returns `x * y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_NegativeOp : MO_ElementWiseUnaryMonoTypeOp<"negative", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Negation operation";
  let description = [{
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_SignedTensor:$input);
}

def MO_PowOp : MO_ElementWiseBinaryOp<"pow", [
  MO_ExpensiveElementwise, Pure]> {
  let summary = "Power operation";
  let description = [{
    Computes `x ** y`, where `x` and `y` are input tensors.

    Examples:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], f32>
      ) : !mo.tensor<[2, 3], f32>

      %x: !mo.tensor<[2, 3], f32>
      %y_int: !mo.tensor<[2, 3], si32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], si32>
      ) : !mo.tensor<[2, 3], f32>
      %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input_x `:` qualified(type($input_x))
    `,` $input_y `:` qualified(type($input_y)) `)` attr-dict
    `:` qualified(type($result))
  }];
}

// TODO(#22958): Consider relaxing constraints on trig. functions
def MO_SinOp : MO_ElementWiseUnaryFloatMonoTypeOp<"sin", [MO_ExpensiveElementwise]> {
  let summary = "Sine operation";
  let description = [{
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}

def MO_CosOp : MO_ElementWiseUnaryFloatMonoTypeOp<"cos", [MO_ExpensiveElementwise]> {
  let summary = "Cosine operation";
  let description = [{
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}

def MO_AtanhOp : MO_ElementWiseUnaryFloatMonoTypeOp<"atanh", [MO_ExpensiveElementwise]> {
  let summary = "Atanh operation";
  let description = [{
    Returns `atanh(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.atanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}


def MO_CumSumOp : MO_TensorOp<"cumsum", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_SingleDevice,
]> {
  let summary = "Cumulative summation operation";
  let description = [{
    Returns the cumulative sum along `axis` (numpy semantics; -1 is the last
    axis). `exclusive` = 1 excludes the current element; `reverse` sums in
    the opposite direction.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.cumsum(%arg) [axis = 0, exclusive = 1, reverse = 0] : (
      !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    IndexAttr:$exclusive,
    IndexAttr:$reverse
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `)`
    `[` `axis` `=` $axis `,` `exclusive` `=` $exclusive `,`
        `reverse` `=` $reverse `]` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_CeilOp : MO_ElementWiseUnaryFloatMonoTypeOp<"ceil", [Idempotent]> {
  let summary = "Ceiling operation";
  let description = [{
    Returns the smallest integer not less than `x`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_FloorOp : MO_ElementWiseUnaryFloatMonoTypeOp<"floor", [Idempotent]> {
  let summary = "Floor operation";
  let description = [{
    Returns the largest integer not greater than `x`.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TruncOp : MO_ElementWiseUnaryFloatMonoTypeOp<"trunc", [Idempotent]> {
  let summary = "Truncation operation";
  let description = [{
    Rounds `x` toward zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RoundOp : MO_ElementWiseUnaryFloatMonoTypeOp<"round", [Idempotent]> {
  let summary = "Rounding operation";
  let description = [{
    Rounds `x` to the nearest integer, ties to even.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryMonoTypeOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_GeluOp : MO_ElementWiseUnaryMonoTypeOp<"gelu", [MO_ExpensiveElementwise]> {
  let summary = "Gaussian error linear unit operation";
  let description = [{
    Returns the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_GeluTanhOp : MO_ElementWiseUnaryMonoTypeOp<"gelu_tanh", [MO_ExpensiveElementwise]> {
  let summary = "Tanh approximation of the GELU operation";
  let description = [{
    Returns the tanh approximation of GELU
    `0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu_tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_GeluQuickOp : MO_ElementWiseUnaryMonoTypeOp<"gelu_quick", [MO_ExpensiveElementwise]> {
  let summary = "Quick (sigmoid) approximation of the GELU operation";
  let description = [{
    Returns the quick approximation of GELU `x * sigmoid(1.702 * x)`, where `x`
    is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.gelu_quick(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SigmoidOp : MO_ElementWiseUnaryMonoTypeOp<"sigmoid", [MO_ExpensiveElementwise]> {
  let summary = "Sigmoid operation";
  let description = [{
    Returns the sigmoid activation `1 / (1 + exp(-x))`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SiluOp : MO_ElementWiseUnaryMonoTypeOp<"silu", [MO_ExpensiveElementwise]> {
  let summary = "SiLU (Swish) operation";
  let description = [{
    Returns the SiLU (Swish) activation `x * sigmoid(x)`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.silu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RsqrtOp : MO_ElementWiseUnaryFloatMonoTypeOp<"rsqrt", [MO_ExpensiveElementwise]> {
  let summary = "Reciprocal square root operation";
  let description = [{
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SqrtOp : MO_ElementWiseUnaryFloatMonoTypeOp<"sqrt", [MO_ExpensiveElementwise]> {
  let summary = "Square root operation";
  let description = [{
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_CastOp : MO_ElementWiseUnaryOp<"cast", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  Pure,
]> {
  let summary = "Casting operation";
  let description = [{
    Returns the input tensor, cast to the specified element type.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], i32>
      %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

def MO_IsNaNOp : MO_ComparisonUnaryOp<"is_nan"> {
  let summary = "NaN test operation";
  let description = [{
    Returns true where `x` is NaN.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_IsInfOp : MO_ComparisonUnaryOp<"is_inf"> {
  let summary = "Infinity test operation";
  let description = [{
    Returns true where `x` is infinite.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_EqualOp : MO_ComparisonBinaryOp<"equal", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Equality test operation";
  let description = [{
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                    !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterOp : MO_ComparisonBinaryOp<"greater", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                      !mo.tensor<[2, 3], f32>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterEqualOp : MO_ComparisonBinaryOp<"greater_equal", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_NotEqualOp : MO_ComparisonBinaryOp<"not_equal", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_AndOp : MO_LogicalBinaryOp<"and"> {
  let summary = "Conjunction operation";
  let description = [{
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_OrOp : MO_LogicalBinaryOp<"or"> {
  let summary = "Disjunction operation";
  let description = [{
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_NotOp : MO_TensorOp<"not", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ElementWiseUnary,
  MO_SameOperandsAndResultType,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Logical negation operation";
  let description = [{
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$input);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_XorOp : MO_LogicalBinaryOp<"xor"> {
  let summary = "Exclusive disjunction operation";
  let description = [{
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

// TODO: Use an AllElementTypesMatch<["x", "y", "result"]> trait for MO_Tensor.
def MO_SelectOp : MO_TensorOp<"select", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ParamDeclaration,
  MO_SameOperandsAndResultShape,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Select operation";
  let description = [{
    Returns `cond ? x : y` elementwise.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$cond, MO_Tensor:$x, MO_Tensor:$y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MO_SliceOp : MO_TensorOp<"slice", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDeviceWithHostOperands<["start", "stop", "step"]>,
]> {
  let summary = "Slice operation";
  let description = [{
    Slices an N-D `input` with numpy semantics: `start`, `stop`, and `step`
    are 1-D index tensors of N elements, and dimension `i` is indexed by
    `start[i]:stop[i]:step[i]`.

    - Negative `start`/`stop` count from the end.
    - Out-of-bound `start`/`stop` are clamped to [-dim, dim].
    - `step` is nonzero and may be negative.

    As in Python, negative indices are normalized (by adding the dimension
    size) before clamping. So `l[:-1:-1]` is empty; reversing `l` takes
    `l[:-N-1:-1]` (or anything below -N-1).

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$start,
    MO_IndicesVectorIfRanked:$stop,
    MO_IndicesVectorIfRanked:$step,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_SubOp : MO_ElementWiseBinaryMonoTypeOp<"sub", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Subtraction operation";
  let description = [{
    Returns `x - y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TanhOp : MO_ElementWiseUnaryFloatMonoTypeOp<"tanh", [MO_ExpensiveElementwise]> {
  let summary = "Hyperbolic tangent operation";
  let description = [{
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Softmax-like ops
//===----------------------------------------------------------------------===//

class MO_SoftmaxLikeOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    AllDTypesMatch<["input", "result"]>,
    AllShapeAttrsMatch<["input", "result"]>,
    MO_ParamDeclaration,
    MO_SingleDevice,
    Pure,
  ]> {
  let arguments = (ins MO_Tensor:$input, IndexAttr:$axis);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `)` `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type($input, results)
  }];
}

def MO_LogSoftmaxOp : MO_SoftmaxLikeOp<"reduce.logsoftmax"> {
  let summary = "Logarithmic softmax operation";
  let description = [{
    Returns `log(softmax(x, axis))`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.logsoftmax(%arg, %axis) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SoftmaxOp : MO_SoftmaxLikeOp<"reduce.softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Returns `exp(x) / sum(exp(x))`, summing along `axis`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.softmax(%arg, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

class MO_ReductionOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    MO_ReductionTraits,
    MO_SingleDevice,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;

  let builders = [
    /// Infers the output type when `outputTy` is null.
    OpBuilder<(ins
      "TensorValue":$input,
      "int64_t":$axis,
      CArg<"TensorType", "{}">:$outputTy
    )>,
  ];

  let extraClassDeclaration = [{
    /// `axis`, made non-negative.
    std::optional<int64_t> getReductionAxis();
    ::mlir::IntegerAttr getAxisAttribute() { return getAxisAttr(); }
    void setAxisAttribute(int64_t axis) {
      setAxisAttr(::mlir::IntegerAttr::get(
          ::mlir::IndexType::get(getContext()), axis));
    }
  }];
}

// Reduction operations where the input and result type match
class MO_MonotypeReductionOp<string name, list<Trait> traits = []> :
  MO_ReductionOp<name, traits # [AllDTypesMatch<["input", "result"]>]>;

// A reduction whose axis is the one input dimension that becomes a static 1
// in the result. When no dimension changes, it's the identity.
// TODO(GEX-3348): delete `MO_ReductionOp` once every reduction is one of
// these, then rename this class to `MO_ReductionOp`.
class MO_ImplicitAxisReductionOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    MO_ReductionTraits,
    MO_SingleDevice,
    AllDTypesMatch<["input", "result"]>,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasFolder = 1;

  let builders = [
    /// Reduces `axis` of `input`, keeping it as size 1.
    OpBuilder<(ins "TensorValue":$input, "int64_t":$axis)>,
  ];

  let extraClassDeclaration = [{
    /// The axis that becomes 1, or none for the identity.
    std::optional<int64_t> getReductionAxis();
    // The axis comes from the result type, not an attribute.
    ::mlir::IntegerAttr getAxisAttribute() { return {}; }
    void setAxisAttribute(int64_t) {}
  }];
}

def MO_ReduceMulOp : MO_MonotypeReductionOp<"reduce.mul", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Product reduction operation";
  let description = [{
    Reduces `input` across `axis` (numpy semantics) to its product,
    keeping that axis as size 1.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.mul(%arg) [axis = 1] : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceAddOp : MO_ImplicitAxisReductionOp<"reduce.add"> {
  let summary = "Sum reduction operation";
  let description = [{
    Reduces `input` to its sum along the one dimension that is a static 1
    in the result.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.add(%arg) : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMinOp : MO_ImplicitAxisReductionOp<"reduce.min"> {
  let summary = "Minimum reduction operation";
  let description = [{
    Reduces `input` to its minimum along the one dimension that is a static 1
    in the result.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.min(%arg) : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMaxOp : MO_ImplicitAxisReductionOp<"reduce.max"> {
  let summary = "Maximum reduction operation";
  let description = [{
    Reduces `input` to its maximum along the one dimension that is a static 1
    in the result.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.reduce.max(%arg) : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Normalization ops
//===----------------------------------------------------------------------===//

def MO_ReduceRMSNormOp : MO_TensorOp<"reduce.rms_norm", [
  AllDTypesMatch<["input", "result"]>,
  AllShapeAttrsMatch<["input", "result"]>,
  MO_ParamDeclaration,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "RMS normalization operation";
  let description = [{
    Computes `input / sqrt(mean(input^2) + epsilon) * (weight + weight_offset)`.

    `epsilon` and `weight_offset` are compile-time constants: targets that
    need scalar constants as attributes cannot take them as operands.

    `multiply_before_cast` false (Llama-style) casts to the output dtype
    before multiplying by the weight; true (Gemma-style) multiplies first.

    Example:

    ```mlir
      %res = mo.reduce.rms_norm(%input, %weight)
        {epsilon = #M.dense_array<9.99999997E-7> : tensor<f32>,
         weight_offset = #M.dense_array<0.0> : tensor<f32>,
         multiply_before_cast = false} :
        (!mo.tensor<[2, 3], bf16, gpu:0>, !mo.tensor<[3], bf16, gpu:0>)
        -> !mo.tensor<[2, 3], bf16, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$weight,
    MO_F32ScalarAttr:$epsilon,
    MO_F32ScalarAttr:$weight_offset,
    BoolAttr:$multiply_before_cast,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `{` `epsilon` `=` $epsilon `,` `weight_offset` `=` $weight_offset `,`
    `multiply_before_cast` `=` $multiply_before_cast `}` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ReduceGroupNormOp : MO_TensorOp<"reduce.group_norm", [
  AllShapeAttrsMatch<["input", "result"]>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["epsilon", "numGroups"]>,
  Pure,
]> {
  let summary = "Group normalization operation";
  let description = [{
    Normalizes with statistics computed per group of channels.

    Example:

    ```mlir
      %res = mo.reduce.group_norm(%input, %gamma, %beta, %epsilon, %num_groups) :
        (!mo.tensor<[1, 32, 64, 64], f32, gpu:0>, !mo.tensor<[32], f32, gpu:0>,
         !mo.tensor<[32], f32, gpu:0>, !mo.tensor<[], f32>, !mo.tensor<[], si32>)
        -> !mo.tensor<[1, 32, 64, 64], f32, gpu:0>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$gamma,
    MO_Tensor:$beta,
    MO_Tensor:$epsilon,
    MO_Tensor:$numGroups,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ReduceMinAndMaxOp : MO_TensorOp<"reduce.reduce_min_and_max", [
  MO_ParamDeclaration,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Combined minimum and maximum reduction operation";
  let description = [{
    Reduces along `axis` to [min, max]: the reduced axis becomes a dimension
    of size 2.

    Example:

    ```mlir
      %res = mo.reduce.reduce_min_and_max(%input) [axis = 1] :
        (!mo.tensor<[2, 10], f32>) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type($input, results)
  }];
}

//===----------------------------------------------------------------------===//
// Sliding window ops
//===----------------------------------------------------------------------===//

def MO_AvgPoolOp : MO_TensorOp<"avg_pool", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
  MO_SlidingWindow,
]> {
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    Supports 2D pooling on 4-D NHWC input.

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. For a
    static-rank input each has `input_rank - 2` elements, one per spatial
    dim, except `paddings`, which has `2 * (input_rank - 2)` laid out as
    (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 = H and
    dim2 = W. Strides and dilations apply to the filter only.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `,`
        `count_boundary` `=` $count_boundary `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_AvgPoolCeilOp : MO_TensorOp<"avg_pool_ceil_mode_true", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
  MO_SlidingWindow,
]> {
  let summary = "Ceiling-mode average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    Supports 2D pooling on 4-D NHWC input.

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. For a
    static-rank input each has `input_rank - 2` elements, one per spatial
    dim, except `paddings`, which has `2 * (input_rank - 2)` laid out as
    (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 = H and
    dim2 = W. Strides and dilations apply to the filter only.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `,`
        `count_boundary` `=` $count_boundary `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolOp : MO_TensorOp<"max_pool", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
  MO_SlidingWindow,
]> {
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    Supports 2D pooling on 4-D NHWC input.

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. For a
    static-rank input each has `input_rank - 2` elements, one per spatial
    dim, except `paddings`, which has `2 * (input_rank - 2)` laid out as
    (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 = H and
    dim2 = W. Strides and dilations apply to the filter only.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<1, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolCeilOp : MO_TensorOp<"max_pool_ceil_mode_true", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
  MO_SlidingWindow,
]> {
  let summary = "Ceiling-mode maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    Supports 2D pooling on 4-D NHWC input.

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. For a
    static-rank input each has `input_rank - 2` elements, one per spatial
    dim, except `paddings`, which has `2 * (input_rank - 2)` laid out as
    (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 = H and
    dim2 = W. Strides and dilations apply to the filter only.

    Example:

    ```mlir
      %fs = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %st = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ConvOp : MO_TensorOp<"conv", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["strides", "dilations", "paddings", "num_groups"]>,
  MO_SlidingWindow,
]> {
  let summary = "Convolution operation";
  let description = [{
    Computes a 1D-3D grouped convolution. Input is channels-last (NHWC for
    2D); the filter is FCRS, i.e. (out_channels, in_channels / num_groups,
    height, width).

    `filter_layout` is the filter's memory layout; if empty, InferLayouts
    picks it (FCRS for 2D, FCQRS for 3D). Supported: FCRS, RSCF (legacy), and
    packed variants like FRSCf.

    `strides`, `dilations`, and `paddings` are rank 1 or unranked. For a
    static-rank input each has `input_rank - 2` elements, one per spatial
    dim, except `paddings`, which has `2 * (input_rank - 2)` laid out as
    (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 = H and
    dim2 = W. Padding is zeros around the spatial dims of `input`; padding a
    2x3 `input` with [0, 1, 2, 1] yields:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    Input, output, and filter ranks must match when static. `num_groups` is a
    ranked scalar dividing both channel counts. Strides and padding apply to
    the input only.

    Example:

    ```mlir
      %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant <#M.dense_array<1> : tensor<si64>> {device = #M.device_ref<"cpu", 0>}
        : %!mo.tensor<[], si64>
      %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[64, 32, 2, 2], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_RankedScalarIndexTensor:$num_groups,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    ` ` `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
                    `paddings` `=` $paddings `,`
                    `num_groups` `=` $num_groups `,`
                    `filter_layout` `=` $filter_layout
                    (`,` `input_layout` `=` $input_layout^)? `]` `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_ConvTransposeOp : MO_TensorOp<"conv_transpose", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["strides", "dilations", "paddings", "output_paddings"]>,
  MO_SlidingWindow,
  MO_ViewLike,
]> {
  let summary = "Transposed convolution operation";
  let description = [{
    Computes the gradient of a convolution with the same filter and
    hyperparameters with respect to its input. See
    https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    Supports 1D-3D spatial dims. With `out_channels` relative to the original
    convolution, input is channels-last (NHWC for 2D, channels =
    out_channels) and the filter is RSFC, i.e. (height, width, out_channels,
    in_channels).

    `strides`, `dilations`, `paddings`, and `output_paddings` are rank 1 or
    unranked. For a static-rank input each has `input_rank - 2` elements, one
    per spatial dim, except `paddings`, which has `2 * (input_rank - 2)` laid
    out as (dim1_before, dim1_after, dim2_before, dim2_after, ...) with dim1 =
    H_out and dim2 = W_out. Padding a 2x4 `output` with [0, 1, 2, 1] yields:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    `output_paddings[i]` appends zeros to output axis `i`, disambiguating the
    output shape when a stride exceeds 1. Only 0 is supported.

    Input, output, and filter ranks must match when static.

    Example:

    ```mlir
      %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[4], si64>
      %op = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> {device = #M.device_ref<"cpu", 0>}
        : !mo.tensor<[2], si64>
      %res = mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ShapeLikeIfRanked:$output_paddings,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    ` ` `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
        `paddings` `=` $paddings `,` `output_paddings` `=` $output_paddings `,`
        `filter_layout` `=` $filter_layout
        (`,` `input_layout` `=` $input_layout^)? `]` `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

class MO_ShapeModifierOp<string mnemonic, list<Trait> traits = []> :
  MO_TensorOp<mnemonic, traits # [
    MO_ParamDeclaration,
    MO_SameFirstOperandAndResultsElementType,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$newShape,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BroadcastToOp : MO_ShapeModifierOp<"broadcast_to", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ParamDeclaration,
  MO_SingleDevice,
  MO_ViewLike,
]> {
  let summary = "Broadcast operation";
  let description = [{
    Broadcasts the input to the given shape with numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %to = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %res = mo.broadcast_to(%from, %to) : (
        !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let builders = [
    /// Broadcasts to the shape of `broadcastLike`. `shapeOfInput` is that
    /// shape, if already available; otherwise a ShapeOfOp is inserted.
    /// Passing it avoids duplicate ops before CSE.
    OpBuilder<(ins
      "TensorValue":$input,
      "TensorValue":$broadcastLike,
      CArg<"TensorValue", "{}">:$shapeOfInput
    )>,
  ];
}

def MO_ReshapeOp : MO_ShapeModifierOp<"reshape", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ExecutesOnHost,
  MO_PreservedDuringKernelLowering,
  MO_ViewLike,
]> {
  let summary = "Reshape operation";
  let description = [{
    Returns the input's data with a new shape. The shape may contain one -1,
    whose dimension is inferred.

    Example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant <#M.dense_array<3, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```

    Auto-sizing example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant <#M.dense_array<3, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_SqueezeShapeOp : MO_ShapeModifierOp<"squeeze_shape", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice,
]> {
  let summary = "Squeeze-shape operation";
  let description = [{
    Computes the shape after a squeeze: removes the `M` indicated dimensions
    from a rank-`N` shape, giving rank `N - M`. Each index must name a
    size-1 dimension (otherwise it is an error) and lie in [-N, N), with
    negative indices counting from the end.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLike:$input_shape,
    MO_ShapeLike:$remove_indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let hasCanonicalizeMethod = 1;
}

def MO_UnsqueezeShapeOp : MO_ShapeModifierOp<"unsqueeze_shape", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice,
]> {
  let summary = "Unsqueeze-shape operation";
  let description = [{
    Computes the shape after an unsqueeze: a rank-`N` shape with `M` indices
    becomes rank `N + M`, with 1 at the (possibly negative) indices.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLike:$input_shape,
    MO_ShapeLike:$padding_indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let hasCanonicalizeMethod = 1;
}

def MO_ShapeOfOp : MO_TensorOp<"shape_of", [
  MO_ExecutesOnHost,
  MO_ParamDeclaration,
]> {
  let summary = "Shape operation";
  let description = [{
    Returns the shape of a tensor.

    Examples:

    ```mlir
      // statically ranked
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape1 = mo.shape_of(%arg1) : !mo.tensor<[3], si64>

      // dynamically ranked
      %arg2: !mo.tensor<?, f32>
      %shape2 = mo.shape_of(%arg2) : !mo.tensor<[?], si64>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "TensorValue":$input)>,
    OpBuilder<(ins "TensorValue":$input, "int64_t":$width)>,
  ];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_BroadcastShapeOp : MO_TensorOp<"broadcast_shape", [
  Commutative,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice,
]> {
  let summary = "Broadcast-shape operation";
  let description = [{
    Returns the shape that two shapes would broadcast to under the numpy rules:

    1. Left-pad the shorter shape with 1s to equal rank.
    2. Each dimension pair must match, or one of them must be 1.

    Examples:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    `shape1` is [4, 2, 3].

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    `shape1` is [10, 2, 5].
  }];

  let arguments = (ins
    MO_ShapeLike:$input_x,
    MO_ShapeLike:$input_y,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

/// Used to model MO ops that accept both MO_Tensor and MOSH_ShapeType.
class MOShape_Op<string mnemonic, list<Trait> traits = []> :
  Op<MODialect, "shape." # mnemonic, traits> {
  string cppNamespace = MODialect.cppNamespace # "::Shape";
}

def MOShape_ToTensorOp : MOShape_Op<"to_tensor", [
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Casting operation";
  let description = [{
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasCanonicalizeMethod = 1;
}

def MOShape_FromTensorOp : MOShape_Op<"from_tensor", [
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Casting operation";
  let description = [{
    Casts the input shape-like tensor to a shape value.

    Example:

    ```mlir
      %sht: !mo.tensor<[2], si64>
      %sh = mo.shape.from_tensor(%sht) : (!mo.tensor<[2], si64>) -> !mosh.ape
    ```
  }];

  let arguments = (ins MO_ShapeLike:$input);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

class MOStatic_TensorOp<string mnemonic, list<Trait> traits = []> :
  MO_TensorOp<"static." # mnemonic, traits> {
  string cppNamespace = MODialect.cppNamespace # "::Static";
}

class MOStatic_ShapeModifierOp<string mnemonic, list<Trait> traits = []> :
  MOStatic_TensorOp<mnemonic, traits # [
    MO_SameFirstOperandAndResultsElementType,
    MO_SingleDevice,
    Pure,
  ]> {
  string cppNamespace = MODialect.cppNamespace # "::Static";

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];
}

def MOStatic_BroadcastToOp : MOStatic_ShapeModifierOp<"broadcast_to", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ViewLike,
]> {
  let summary = "Broadcasting operation";
  let description = [{
    Broadcasts the input tensor to the result type with numpy semantics. Input
    and result shapes must be fully known, but can be parametric.

    Compile-time validation of the target shape is limited and runtime checks
    are the user's job, so frontends should use `mo.broadcast_to` with a
    constant shape-like tensor instead.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %res1 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
      kgen.param.declare N = <...>
      %res2 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MOStatic_ReshapeOp : MOStatic_ShapeModifierOp<"reshape", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ViewLike,
]> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same underlying data but a different shape. Input
    and result shapes must be fully known, but can be parametric; inferred
    dimensions (-1 in `mo.reshape`) are not allowed.

    The target shape is not validated at compile time or runtime, so frontends
    should use `mo.reshape` with a constant shape-like tensor instead.

    Example:

    ```mlir
      %from: !mo.tensor<[2, 3], f32>
      %res = mo.static.reshape(%from)
        : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_ConcatOp : MO_TensorOp<"concat", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice,
]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates one or more `inputs` along `axis` (numpy semantics, -1 is the
    last axis). Inputs and output share the element type and all dimensions
    except `axis`, whose output size is the sum of the input sizes.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %res = mo.concat[1](%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    IndexAttr:$axis,
    Variadic<MO_Tensor>:$inputs,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_SplitOp : MO_TensorOp<"split", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["splitSizes"]>,
]> {
  let summary = "Splitting operation";
  let description = [{
    Splits `input` along `axis` (numpy semantics, -1 is the last axis) into one
    output per element of the rank-1 `splitSizes`. Each output matches `input`
    except along `axis`, where its size is the corresponding `splitSizes`
    element.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 8], f32>
      %splitSizes = mo.constant <#M.dense_array<3, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
      %res:2 = mo.split(%input, %splitSizes) [axis = 1] : (
        !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
      ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$splitSizes,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `(` $input `,` $splitSizes `)` `[` `axis` `=` $axis `]` attr-dict `:` `(`
      qualified(type($input)) `,` qualified(type($splitSizes))
    `)` `->` `(` type($results) `)`
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_TransposeOp : MO_TensorOp<"transpose", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["perm"]>,
]> {
  let summary = "Transposition operation";
  let description = [{
    Permutes the dimensions of `input` by `perm`, which must contain unique
    values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$perm,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MO_TileOp : MO_TensorOp<"tile", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["repeats"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Tiling operation";
  let description = [{
    Repeats the input `repeats[i]` times along each dimension `i`, so output
    dimension `i` is `input.dim(i) * repeats[i]`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$repeats,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MO_PadConstantOp : MO_TensorOp<"pad.constant", [
  MO_PadLikeTraits,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["paddings", "constant"]>,
]> {
  let summary = "Constant-mode padding operation";
  let description = [{
    Pads `input` with the scalar `constant`. For a rank-`N` input, `paddings`
    has shape `(2 * N)` holding a `[before, after]` pair per dimension, so
    output dimension D is `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant <#M.dense_array<1.0> : tensor<f32>> : !mo.tensor<[], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_RankedScalarTensor:$constant,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_PadReflectOp : MO_TensorOp<"pad.reflect", [
  MO_SingleDeviceWithHostOperands<["paddings"]>,
  MO_PadLikeTraits,
  MO_ParamDeclaration,
]> {
  let summary = "Reflection-mode padding operation";
  let description = [{
    Pads `input` by reflection. `paddings` is laid out as in `mo.pad.constant`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_PadRepeatOp : MO_TensorOp<"pad.repeat", [
  MO_SingleDeviceWithHostOperands<["paddings"]>,
  MO_PadLikeTraits,
  MO_ParamDeclaration,
]> {
  let summary = "Repeat-mode padding operation";
  let description = [{
    Pads `input` by repeating border values. `paddings` is laid out as in
    `mo.pad.constant`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant <#M.dense_array<1, 0, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def MO_GatherOp : MO_TensorOp<"gather", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice,
]> {
  let summary = "`gather` operation";
  let description = [{
    Gathers slices from input's axis according to indices. For static ranks,
    the output rank is `inputRank - 1 + indicesRank`:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    `axis` and `indices` follow numpy semantics (-1 is the last axis).

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %res = mo.gather(%input, %indices) [axis = 0] : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins
    MO_NonScalarIfRanked:$input,
    MO_IndicesTensor:$indices,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

// TODO(#20442): Remove with generic gather fusion.
def MO_GatherSumOp : MO_TensorOp<"gather_sum", [
  MO_SingleDevice,
  AllDTypesMatch<["input", "result"]>,
]> {
  let summary = "`gather-sum` operation";
  let description = [{
    Gather on axis 0 followed by a sum reduction on axis 1.
  }];

  let arguments = (ins
    MO_Rank2Tensor:$input,
    MO_Rank2SI32Tensor:$indices
  );
  let results = (outs MO_Tensor:$result);

  let hasVerifier = 1;
}

def MO_GatherNdOp : MO_TensorOp<"gather_nd", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDevice,
]> {
  let summary = "Multi-dimensional `gather` operation";
  let description = [{
    Variant of `mo.gather` with multi-dimensional indices. The last indices
    dimension holds the index (at most input rank long; shorter gathers
    leading-dimension slices) and the outer dimensions act as batch dimensions.

    ```
    output_shape = (
          input.shape[:batch_dims]
        + indices.shape[batch_dims:-1]
        + data.shape[batch_dims + indices.shape[-1]:]
    )
    ```

    ```mlir
      %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<2x2x4xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> {device = #M.device_ref<"cpu", 0>} :
        !mo.tensor<[3], si64>

      %result = mo.gather_nd(%input, %indices) [batchDims = 0] :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins
    MO_NonScalarIfRanked:$input,
    MO_IndicesTensor:$indices,
    // TODO(#21324): Support dynamic batchDims
    IndexAttr:$batchDims,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `batchDims` `=` $batchDims `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def MO_ScatterOp : MO_TensorOp<"scatter", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDevice,
]> {
  let summary = "`scatter` operation";
  let description = [{
    Returns a copy of `input` with elements from `updates` written at positions
    given by `indices`. `input`, `updates`, and `indices` have the same rank;
    each `indices` entry replaces the `axis` coordinate of its own index. In
    2D:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter(%inputs, %updates, %indices) [axis = 0] : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

class MO_ScatterReductionOp<string name, list<Trait> traits = []> :
  MO_TensorOp<"scatter." # name, traits # [
    MO_SingleDeviceWithHostOperands<["axis"]>,
    MO_ScatterLikeTraits,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterAddOp : MO_ScatterReductionOp<"add"> {
  let summary = "`scatter-add` operation";
  let description = [{
    Like `mo.scatter`, but combines each update with the existing element via
    addition. In 2D:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMaxOp : MO_ScatterReductionOp<"max"> {
  let summary = "scatter-max operation";
  let description = [{
    Like `mo.scatter`, but combines each update with the existing element via
    maximum. In 2D:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMinOp : MO_ScatterReductionOp<"min"> {
  let summary = "`scatter-min` operation";
  let description = [{
    Like `mo.scatter`, but combines each update with the existing element via
    minimum. In 2D:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMulOp : MO_ScatterReductionOp<"mul"> {
  let summary = "scatter-mul operation";
  let description = [{
    Like `mo.scatter`, but combines each update with the existing element via
    multiplication. In 2D:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

def MO_ScatterNdOp : MO_TensorOp<"scatter_nd", [
  MO_ScatterNdLikeTraits,
  MO_SingleDevice,
]> {
  let summary = "Multi-dimensional `scatter` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    The last dimension of `indices` is an index vector into a copy of `input`;
    the addressed slice (or scalar, if the vector length equals the input rank)
    is replaced by the corresponding slice of `updates`.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

class MO_ScatterNdReductionOp<string name, list<Trait> traits = []> :
  MO_TensorOp<"scatter_nd." # name, traits # [
    MO_ScatterNdLikeTraits,
    MO_SingleDevice,
  ]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_NonScalarIfRanked:$updates,
    MO_IndicesTensor:$indices,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterNdAddOp : MO_ScatterNdReductionOp<"add"> {
  let summary = "Multi-dimensional `scatter-add` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Index layout is as in `mo.scatter_nd`.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMaxOp : MO_ScatterNdReductionOp<"max"> {
  let summary = "Multi-dimensional `scatter-max` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Index layout is as in `mo.scatter_nd`.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMinOp : MO_ScatterNdReductionOp<"min"> {
  let summary = "Multi-dimensional `scatter-min` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Index layout is as in `mo.scatter_nd`.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMulOp : MO_ScatterNdReductionOp<"mul"> {
  let summary = "Multi-dimensional `scatter-mul` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Index layout is as in `mo.scatter_nd`.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NonMaximumSuppressionOp
//===----------------------------------------------------------------------===//

def MO_NonMaximumSuppressionOp : MO_TensorOp<"non_maximum_suppression", [
  MO_SingleDeviceWithHostOperands<["maxOutputBoxesPerClass", "iouThreshold", "scoreThreshold"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Non-max suppression operation";
  let description = [{
   Filters out boxes that have high intersection-over-union (IOU).

   `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
   coordinates of any diagonal pair of box corners and the coordinates can be
   provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

    Example:

    ```mlir
      %boxes : !mo.tensor<[1, 6, 4], f32>
      %scores : !mo.tensor<[1, 1, 6], f32>
      %maxOutputBoxesPerClass : !mo.tensor<[], si64>
      %iouThreshold : !mo.tensor<[], si64>
      %scoreThreshold : !mo.tensor<[], si64>
      %res = mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    ```
  }];
  // TODO(#21506): Allow any ranked tensors as inputs.
  let arguments = (ins
    MO_Rank3Tensor:$boxes,
    MO_Rank3Tensor:$scores,
    MO_RankedScalarIndexTensor:$maxOutputBoxesPerClass,
    MO_RankedScalarFloatTensor:$iouThreshold,
    MO_RankedScalarFloatTensor:$scoreThreshold,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Rank2IndexTensor:$output);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

def MO_ArgNonzeroOp : MO_TensorOp<"arg_nonzero", [
  MO_SingleDevice,
  MO_ParamDeclaration,
]> {
  let summary = "Nonzero-value indices operation";
  let description = [{
    Returns the row-major coordinates of the input's nonzero values as a
    `[nnz, rank_in]` tensor.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>> : !mo.tensor<[3, 3], si32>
      %1 = mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_I64Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ArgMaxOp / ArgMinOp
//===----------------------------------------------------------------------===//

class MO_ArgMinMaxOp<string mnemonic, list<Trait> traits = []> :
  MO_ReductionOp<mnemonic, traits> {
  let arguments = (ins
    MO_Tensor:$input,
    IndexAttr:$axis,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_IndicesTensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `axis` `=` $axis `]` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MO_ArgMaxOp : MO_ArgMinMaxOp<"reduce.arg_max"> {
  let summary = "Maximum-value indices operation";
  let description = [{
    Like `reduce_max` over `axis` (numpy semantics; reduced axis kept as size
    1), but returns indices. Ties return the lowest index.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
      %1 = mo.arg_max(%0) [axis = 1] : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_ArgMinOp : MO_ArgMinMaxOp<"reduce.arg_min"> {
  let summary = "Minimum-value indices operation";
  let description = [{
    Like `reduce_min` over `axis` (numpy semantics; reduced axis kept as size
    1), but returns indices. Ties return the lowest index.

    Example:

    ```mlir
      %0 = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>> : !mo.tensor<[2, 2], si32>
      %1 = mo.arg_min(%0) [axis = 1] : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

def MO_RandomNormalOp : MO_TensorOp<"random.normal", [
  AllDTypesMatch<["mean", "variance"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["shape", "mean", "variance"]>,
]> {
  let summary = "Normal distribution operation";
  let description = [{
    Returns a `shape` tensor of normal samples with mean `mean` and standard
    deviation `variance`.

  Example:
    ```mlir
      %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %mean = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
      %variance = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
      %seed = mo.constant <#M.dense_array<1> : tensor<1xui64>> : !mo.tensor<[1], ui64>
      %res = mo.random.normal(%size, %mean, %variance, %seed) :
            (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
            !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$shape,
    MO_RankedScalarFloatTensor:$mean,
    MO_RankedScalarFloatTensor:$variance,
    MO_Rank1IndexTensor:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

def MO_RandomUniformOp : MO_TensorOp<"random.uniform", [
  AllDTypesMatch<["lowerBound", "upperBound", "result"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["shape", "lowerBound", "upperBound"]>,
]> {
  let summary = "Uniform distribution operation";
  let description = [{
    Returns a `shape` tensor of uniform samples from [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %lowerBound = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[], f32>
    %upperBound = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[], f32>
    %seed = mo.constant <#M.dense_array<1> : tensor<1xui64>> : !mo.tensor<[1], ui64>
    %res = mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$shape,
    MO_RankedScalarTensor:$lowerBound,
    MO_RankedScalarTensor:$upperBound,
    MO_Rank1IndexTensor:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

def MO_RangeOp : MO_TensorOp<"range", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ParamDeclaration,
  MO_SameOperandsAndResultElementType,
  MO_SingleDeviceWithHostOperands<["start", "limit", "step"]>,
]> {
  let summary = "Range operation";
  let description = [{
    Returns `[start, limit)` in increments of `step`. All arguments share an
    element type. `step` must be non-zero and `limit - start` must be zero or
    have the sign of `step`.

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
      %step = mo.constant <#M.dense_array<1.5> : tensor<f32>> : !mo.tensor<[], f32>
      %res = mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
      %stepInt = mo.constant <#M.dense_array<2> : tensor<si32>> : !mo.tensor<[], si32>
      %limitInt = mo.constant <#M.dense_array<11> : tensor<si32>> : !mo.tensor<[], si32>
      %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarTensor:$start,
    MO_RankedScalarTensor:$limit,
    MO_RankedScalarTensor:$step,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Resize op
//===----------------------------------------------------------------------===//


def MO_ResizeLinearOp : MO_TensorOp<"resize.linear", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["size"]>,
  MO_ParamDeclaration,
]> {
  let summary = "Linear resize operation";
  let description = [{
    Resizes a tensor with linear interpolation. `antialias` applies an
    antialiasing filter when downscaling.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    BoolAttr:$antialias,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `coordinate_transform_mode` `=`
            qualified($coordinate_transform_mode) `,`
        `antialias` `=` $antialias `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ResizeNearestOp : MO_TensorOp<"resize.nearest", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["size"]>,
]> {
  let summary = "Nearest-neighbor resize operation";
  let description = [{
    Resizes a tensor with nearest-neighbor interpolation. `round_mode` is
    0 = HalfDown (default), 1 = HalfUp, 2 = Floor, 3 = Ceil.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant <#M.dense_array<1, 1, 7, 8> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.resize.nearest(%input, %size)
        [coordinate_transform_mode = #mo.coord_trans_mode<half_pixel>,
         round_mode = 2] :
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    DefaultValuedAttr<I64Attr, "0">:$round_mode,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `coordinate_transform_mode` `=`
            qualified($coordinate_transform_mode)
        (`,` `round_mode` `=` $round_mode^)? `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ResizeBicubicOp : MO_TensorOp<"resize.bicubic", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["size"]>,
]> {
  let summary = "Bicubic resize operation";
  let description = [{
    Resizes a tensor with bicubic interpolation (Keys' cubic convolution,
    a = -0.5, over a 4x4 neighborhood).

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant <#M.dense_array<1, 3, 448, 448> : tensor<4xsi64>> : !mo.tensor<[4], si64>
      %res = mo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ROI align op
//===----------------------------------------------------------------------===//

def MO_RoiAlignOp : MO_TensorOp<"roi_align", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["output_height", "output_width", "spatial_scale", "sampling_ratio"]>,
]> {
  let summary = "ROI alignment operation";
  let description = [{
    Pools `input` over each region of interest in `rois`.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant <#M.dense_array<5> : tensor<1xsi64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], f32>

      %res = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
        [aligned = false, mode = "AVG"]
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```

  }];
  let arguments = (ins
    MO_Rank4Tensor:$input,
    MO_Rank2Tensor:$rois,
    MO_RankedScalarIndexTensor:$output_height,
    MO_RankedScalarIndexTensor:$output_width,
    MO_RankedScalarFloatTensor:$spatial_scale,
    MO_RankedScalarFloatTensor:$sampling_ratio,
    BoolAttr:$aligned,
    Builtin_StringAttr:$mode,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Rank4Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    `[` `aligned` `=` $aligned `,` `mode` `=` $mode `]` attr-dict `:`
    functional-type(operands, results)
  }];
  // TODO: add verifier that checks that 2nd rois dimension is 4
}

//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//

class MO_DotBase<string name, list<Trait> traits = []> :
  MO_TensorOp<name, traits # [
    MO_MatmulLike,
    MO_ParamDeclaration,
    MO_SingleDevice,
    Pure,
  ]> {
  let description = [{
    Base class for two-input tensor contractions such as matmul.
  }];
}

def MO_MatmulOp : MO_DotBase<"matmul"> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two 2D tensors.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[10, 20], f32>
      %rhs: ... !mo.tensor<[20, 5], f32>
      %res = mo.matmul(%lhs, %rhs) : (
        !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
      ) -> !mo.tensor<[10, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<BoolAttr, "false">:$packed_b
  );
  let results = (outs MO_Rank2Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` `` custom<MatmulAttrs>($transpose_b, $packed_b) attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_BatchMatmulOp : MO_TensorOp<"batch_matmul", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_MatmulLike,
  MO_ParamDeclaration,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Batched matrix-matrix multiplication operation";
  let description = [{
    Multiplies two batches of matrices; the last two dimensions of each input
    are the matrix dimensions.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[3, 4, 5], f32>
      %rhs: ... !mo.tensor<[3, 5, 6], f32>
      %res = mo.batch_matmul(%lhs, %rhs) :
        mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
      ) -> !mo.tensor<[3, 4, 6], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input_a,
    MO_Tensor:$input_b,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Create matmul with default attribute flags off.
    OpBuilder<(ins
      "Type":$type,
      "Value":$input_a,
      "Value":$input_b,
      "ArrayRef<KGEN::ParamDeclAttr>":$decls
    )>,
  ];

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)`
    (` ` `[` `transpose_b` `=` $transpose_b^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Linear Algebra ops
//===----------------------------------------------------------------------===//

class MO_LinalgOp<string mnemonic, list<Trait> traits = []> :
  MO_TensorOp<"linalg." # mnemonic, traits # [Pure]> {
  string cppNamespace = MODialect.cppNamespace # "::LinAlg";
}

// TODO: Use an AllElementTypesMatch<["input", "result"]> trait for MO_Tensor.
def MO_BandPartOp : MO_LinalgOp<"band_part", [
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["num_lower", "num_upper", "exclude"]>,
]> {
  let summary = "Matrix band part operation";
  let description = [{
    Zeroes everything outside the central band of the matrices formed by the
    last two axes (leading axes are batches):

    out[..., m, n] = in_band(m, n) * input[..., m, n]
    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    A negative bound keeps the whole lower/upper triangle. `exclude` inverts
    the selection, zeroing the band instead.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[], si64>
      %num_upper = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
      %exclude = mo.constant <#M.dense_array<0> : tensor<1xui8>> : !mo.tensor<[], bool>
      %res = mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$num_lower,
    MO_RankedScalarIndexTensor:$num_upper,
    MO_CondLike:$exclude,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape function invocation
//===----------------------------------------------------------------------===//

def MO_InvokeShapeFuncOp : MO_TensorOp<"invoke_shape_func", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExecutesOnHost,
]> {
  let summary = "Shape function call operation";
  let description = [{
    Invokes the named shape function and binds its results to the output
    parameters. Arguments must be shape-like or integer scalar tensors.
    `kgenParams` is forwarded to the shape function as is.

    Example:

    ```mlir
      %input : !mo.tensor<[D1, D2], f32>
      %axis  : !mo.tensor<[], si64>
      %input_shape = mo.shape_of(%input)
      : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si64>

      mo.invoke_shape_func["shape_func_name"]<() -> D1, D2>(%input_shape, %axis)
      : (!mo.tensor<[2], si64>, !mo.tensor<[], si64>)
    ```

    `dataDeptTensors` lists input indices whose values (not just shapes) the
    shape function reads, such as a dynamic reduction axis or slice steps.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque]>>:$inputs,
    StrAttr:$shapeFuncName,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<DictionaryAttr>:$kgenParams,
    OptionalAttr<ArrayAttr>:$dataDeptTensors
  );

  let assemblyFormat = [{
    `[` $shapeFuncName `]` `` custom<ParamDecls>($outputParamDecls)
    `(` $inputs `)` `` custom<ShapeFuncAttrs>($kgenParams, $dataDeptTensors)
    attr-dict `:` `(` type($inputs) `)`
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layout Transform
//===----------------------------------------------------------------------===//

def MO_LayoutTransformOp : MO_TensorOp<"layout.transform", [MO_HostOnly]> {
  let summary = "Layout transformation operation";
  let description = [{
    Transforms the input to the layout of the result type. Both types must carry
    layout annotations. `kgenParams` is forwarded as named parameters to the
    Mojo kernel.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    OptionalAttr<DictionaryAttr>:$kgenParams
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` (`[` `kgenParams` `=` $kgenParams^ `]`)? attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Temp layer norm ops
//===----------------------------------------------------------------------===//

def MO_LayerNormOp : MO_TensorOp<"reduce.layer_norm", [
  MO_ParamDeclaration,
  MO_SingleDeviceWithHostOperands<["epsilon"]>,
]> {
  let summary = "Layer normalization operation";
  let description = [{
    Normalizes over the last dimension of `input`; `gamma` and `beta` have
    shape [channels], the size of that dimension:

      result = (input - mean(input)) / sqrt(var(input) + epsilon) * gamma + beta
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$gamma,
    MO_Tensor:$beta,
    MO_Tensor:$epsilon,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

def MO_DebugPrintOp : MO_TensorOp<"debug.print", [
  MO_ExecutesOnHost,
  MO_PreservedDuringKernelLowering,
  MemoryEffects<[MemWrite]>,
]> {
  let summary = "Printing operation";
  let description = [{
    Prints a debug string, prefixed with `label` if set. For debugging and
    testing only.

    Example:
    ```mlir
      %ch0: !mo.chain
      %ch1 = mo.debug.print(%ch0) [value = "message", label = "label"]
    ```
  }];

  let arguments = (ins
    MO_Chain:$inChain,
    StrAttr:$value,
    DefaultValuedAttr<StrAttr, "\"\"">:$label
  );
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)` `[` `value` `=` $value (`,` `label` `=` $label^)? `]`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

def MO_DebugTensorPrintOp : MO_TensorOp<"debug.tensor.print", [
  MO_PreservedDuringKernelLowering,
  MO_SingleDeviceWithHostOperands<[], ["input"]>,
  MemoryEffects<[MemWrite]>,
]> {
  let summary = "Printing operation";
  let description = [{
    Prints `input`: its contents with `label` if set, otherwise only its
    metadata. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      %ch0: !mo.chain
      %ch1 = mo.debug.tensor.print(%ch0, %arg) [label = "label"] : !mo.tensor<[5], f32>
    ```
  }];

  let arguments = (ins
    MO_Chain:$inChain,
    MO_Tensor:$input,
    DefaultValuedAttr<StrAttr, "\"\"">:$label
  );
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)` (`[` `label` `=` $label^ `]`)? attr-dict `:`
    qualified(type($input))
  }];
}

class MO_SelectKLikeOp<string mnemonic, list<Trait> traits = []> :
  MO_TensorOp<mnemonic, [
    DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
    MO_ParamDeclaration,
    MO_SingleDeviceWithHostOperands<["k", "axis", "sorted"]>,
  ] # traits> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$K,
    MO_RankedScalarIndexTensor:$axis,
    MO_CondLike:$sorted,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs
    MO_Tensor:$values,
    MO_I64Tensor:$indices
  );
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}


def MO_TopKOp : MO_SelectKLikeOp<"top_k"> {
  let summary = "Top-K operation";
  let description = [{
    Returns the `k` largest values along `axis` (in [-rank, rank)) and their
    indices. Returned values are always sorted (stable).

    Example:
    ```mlir
      %in = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>> : !mo.tensor<[2, 6], si64>
      %k = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
      %axis = mo.constant <#M.dense_array<1> : tensor<si64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %sorted = mo.constant <#M.dense_array<1> : tensor<1xi1>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], bool>
      %values, %indices = mo.top_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

def MO_BottomKOp : MO_SelectKLikeOp<"bottom_k"> {
  let summary = "Bottom-K operation";
  let description = [{
    Returns the `k` smallest values along `axis` and their indices. Returned
    values are always sorted (stable).

    Example:
    ```mlir
      %in = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>> : !mo.tensor<[2, 6], si64>
      %k = mo.constant <#M.dense_array<3> : tensor<si64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %axis = mo.constant <#M.dense_array<1> : tensor<si64>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], si64>
      %sorted = mo.constant <#M.dense_array<1> : tensor<1xi1>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[], bool>
      %values, %indices = mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

def MO_CustomOp : MO_TensorOp<"custom", [
  DeclareOpInterfaceMethods<MO_ConditionallyInPlaceInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, [
    "getInChains",
    "getInChainsMutable",
    "getOutChains",
  ]>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_ExplicitDevice,
  MO_ParamDeclaration,
]> {
  let summary = "User-defined operation";
  let description = [{
    `symbol` names the Mojo kernel implementing the op; the kernel must be
    registered under exactly that string.

    Example:

    ```mlir
      %0 = mo.custom [symbol = "test_custom_op"](%arg0)
        (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
    ```

    Corresponding kernel definition:

    ```mojo
      @register_internal("test_custom_op")
      def foo(...):
        pass
    ```

    A kernel that mutates a `mo.buffer` (NDBuffer-based kernels only) needs no
    change; the op takes the buffer operand, an input chain as its final
    operand, and produces an output chain as its final result.

    Example:

    ```mlir
    %ch1 = mo.custom [symbol = "test_mutable_op"](%arg0, %ch0) :
      (!mo.buffer<[D1, D2], f32>, !mo.chain) -> !mo.chain
    ```
    ```mojo
      @register_internal("test_custom_op")
      def foo(...):
        pass
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    StrAttr:$symbol,
    DefaultValuedAttr<M_DeviceRefAttr, "\"cpu\", 0">:$device,
    DefaultValuedAttr<DictionaryAttr, "{}">:$parameters,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
       `` custom<CustomAttrs>($symbol, $device, $parameters)
       attr-dict `(` operands `)` `:`
       functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    std::optional<ChainValue> maybeGetInChain();
    std::optional<ChainValue> maybeGetOutChain();
    mlir::SmallVector<Value> getBufferOperands();
  }];
}

def MO_AssertOp : MO_Op<"assert"> {
  let summary = "Assert that a single boolean value is true.";

  let description = [{
    Bare asserts are not used in MO graphs; they are wrapped in a KGEN apply
    attr and materialized via `kgen.param.declare`. To guard an op:

    1. Create each assert with `emitStaticCall`.
    2. Pass the resulting param-declare ops to `materializeAllAssertLogic`,
       which chains them and guards the op's operands on that chain.

    Example:

    ```mlir
    // When using emitStaticCall and materializeAllAssertLogic
    %X: mo.tensor<[D1, D2]>,
    %Y: mo.tensor<[D3, D4]>

    ...

    kgen.param.declare CH0:
      <apply(:(i1) -> !mo.chain "mo.assert"{message = "Error 1" : !kgen.string}, eq(D1, D3))>

    kgen.param.declare CH1:
      <apply(:(!mo.chain, i1) -> !mo.chain "mo.assert"{message = "Error 2" : !kgen.string}, CH0, eq(D2, D4))>

    %chain1 = mosh.param.to_value = <CH0>
    %chain2 = mosh.param.to_value = <CH1>

    %guard_chain = mo.chain.create(%chain1, %chain2)

    %Xp, Yp = mo.guard[%guard_chain](%X, %Y)

    ...

    %Z = mo.op(%Xp, %Yp)
    ```
  }];

  let arguments = (ins
    Optional<MO_Chain>:$inChain,
    BoolScalar:$cond,
    AttrOfType<KGEN_StringType>:$message
  );
  let results = (outs MO_Chain:$chain);

  let builders = [
    OpBuilder<(ins "Value":$cond, "const llvm::Twine &":$message)>,
  ];

  let assemblyFormat = "(`[`$inChain^`]`)? $cond `,` custom<StringParam>($message) attr-dict";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns apply(mo.assert, cond, message).
    static TypedAttr emitStaticCall(TypedAttr cond, const Twine &message);

    /// Chains `asserts` together, joins their chains with mo.chain.create, and
    /// routes all of `op`'s operands through an mo.guard on that chain.
    static void materializeAllAssertLogic(mlir::RewriterBase &rewriter,
                                          Operation *op,
                                          ArrayRef<KGEN::ParamDeclareOp> asserts);
  }];
}

def MO_GuardOp : MO_Op<"guard", [
  AllTypesMatch<["inputs", "results"]>,
  InferTypeOpAdaptor,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "SSA-value guard operation";
  let description = [{
    Copies `inputs`, ordered after `chain`. Makes downstream ops depend on a
    chain-producing op such as mo.assert.

    Example:

    ```mlir
      %X: !mo.tensor<[D1, D2], f32>
      %Y: !mo.tensor<[D2, D3], f32>

      %chain : !mo.chain

      %guarded:2 = mo.guard[%chain](%X, %Y)

      %z = mo.matmul(%guarded#0, %guarded#1)
    ```
  }];

  let arguments = (ins
    MO_Chain:$chain,
    Variadic<MO_AnyNonBufferType>:$inputs
  );

  let results = (outs Variadic<MO_AnyNonBufferType>:$results);

  let assemblyFormat = [{
    `[` $chain `]` `(` $inputs `)` attr-dict `:` type($inputs)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def MO_ChainCreateOp : MO_Op<"chain.create", [
  Commutative,
  MO_ExecutesOnHost,
  MO_HasDeviceInterface,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Create a new chain conditioned on a variadic number of inputs";
  let description = [{
    Produces a chain that depends on all `inputs`, for sequencing a
    side-effecting op after several producers.

    ```mlir

    %ch0 = mo.assert ...
    %x = mo.matmul ....

    %ch1 = mo.chain.create(%ch0: !mo.chain, %x: mo.tensor)
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs MO_Chain:$result);
  let hasCustomAssemblyFormat = 1;

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Buffer ops
//===----------------------------------------------------------------------===//

def MO_BufferTransferOp : MO_DeviceAgnosticMutableOp<"buffer.transfer", [
  AllDTypesMatch<["src", "dst"]>,
  AllShapeAttrsMatch<["src", "dst"]>,
  MO_PreservedDuringKernelLowering,
]> {
  let summary = "Transfer data between two buffers";
  let description = [{
    Copies `src` into `dst`, which must have the same shape and dtype.

    Example:
    ```mlir
    %outChain = mo.buffer.transfer[%inChain](%src, %dst) : !mo.buffer<[2,3], f32, gpu:0>, !mo.buffer<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Buffer:$src,
    MO_Buffer:$dst,
    MO_Chain:$inChain
  );
  let results = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($src)) `,` qualified(type($dst))
  }];
}


// TODO(GEX-1292): Support parameterization interfaces for this op. Not needed
// yet: it is only created by a pass that copies its shape from an existing
// mo.buffer.
def MO_BufferCreateOp : MO_Op<"buffer.create", [
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Creates a buffer on a specific device, optionally initialized.";
  let description = [{
    Creates a buffer of the result type. By default it is uninitialized and
    reallocated on every execution.

    With `initValue` (same element type as the buffer), the buffer is
    persistent: allocated and filled once during the model's `init` phase,
    then reused across `execute` calls. This gives kernels scratch state that
    survives calls, such as a counter the kernel re-zeroes after each run.

    Example:
    ```mlir
    %buf = mo.buffer.create : !mo.buffer<[20, 20], f32, gpu:0>
    %zeroed = mo.buffer.create [initValue = 0.0 : f32] : !mo.buffer<[20, 20], f32, gpu:0>
    ```
  }];

  let arguments = (ins
    // Single-element elements attr of the buffer's dtype (verified).
    OptionalAttr<AnyAttr>:$initValue
  );
  let results = (outs MO_Buffer:$result);

  let assemblyFormat = [{
    (` ` `[` `initValue` `=` $initValue^ `]`)? attr-dict `:`
    qualified(type($result))
  }];
  let hasVerifier = 1;

  // The `{}` default lets Python bindings omit `initValue`; MLIR attribute
  // casters reject `None`.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$result,
                   CArg<"::mlir::Attribute", "{}">:$initValue)>,
  ];
}

def MO_MutableLoadOp : MO_MutableOp<"mutable.load", [
  AllDTypesMatch<["inBuffer", "outTensor"]>,
  AllShapeAttrsMatch<["inBuffer", "outTensor"]>,
  // outTensor is a result; the trait constrains its device by default.
  MO_PreservedDuringKernelLowering,
  MO_SingleDeviceWithHostOperands</*hostOperands*/[], /*deviceOperands*/["inBuffer"]>,
]> {
  let summary = "Take a chain and a buffer semantic tensor and returns a chain and a value semantic tensor.";
  let description = [{
    Copies `inBuffer` to `outTensor`; with mo.mutable.store, models in-place
    ops. The output chain has at most one use. If `outTensor` has more than
    one use, the op cannot fuse with compute ops.
  }];

  let arguments = (ins MO_Buffer:$inBuffer, MO_Chain:$inChain);
  let results = (outs MO_Tensor:$outTensor, MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_MutableStoreOp : MO_MutableOp<"mutable.store", [
  AllDTypesMatch<["inBuffer", "inTensor"]>,
  AllShapeAttrsMatch<["inBuffer", "inTensor"]>,
  MO_SingleDeviceWithHostOperands<
    /*hostOperands*/[],
    /*deviceOperands*/["inBuffer", "inTensor"]>,
]> {
  let summary = "Takes a chain, a buffer semantic tensor, and a value semantic tensor and returns a chain";
  let description = [{
    Copies `inTensor` to `inBuffer`; with mo.mutable.load, models in-place
    ops. The output chain has at most one use. If `inTensor` has more than
    one use, the op cannot fuse with compute ops.
  }];

  let arguments = (ins
    MO_Buffer:$inBuffer,
    MO_Tensor:$inTensor,
    MO_Chain:$inChain
  );
  let results = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_MutableStoreSliceOp : MO_MutableOp<"mutable.store.slice", [
  MO_SingleDeviceWithHostOperands<
    /*hostOperands*/["start", "stop", "step"],
    /*deviceOperands*/["inBuffer", "slice"]>,
]> {
  let summary = "Stores a tensor to a slice of a buffer semantic tensor.";
  let description = [{
    Stores `slice` into the elements of `inBuffer` selected by the per-dimension
    `start`, `stop`, and `step` vectors, with the same indexing semantics as
    `mo.slice`.

    Example:
    ```mlir
      %buffer: !mo.buffer<[20, 20], f32>
      %slice: !mo.tensor<[D0, D1], f32>
      %ch: !mo.chain

      %start: !mo.tensor<[2], si64> // [1,  -6]
      %stop: !mo.tensor<[2], si32>  // [-3, -3]
      %step: !mo.tensor<[2], si64>  // [5,   1]

      // equivalent to this in numpy: `buffer[1:-3:5, -6:-3:1] = slice`
      %ch' = mo.mutable.store.slice(%ch, %buffer, %slice, %start, %stop, %step)
    ```

    The output chain has at most one use.
  }];
  let arguments = (ins
    MO_Buffer:$inBuffer,
    MO_Tensor:$slice,
    MO_IndicesVectorIfRanked:$start,
    MO_IndicesVectorIfRanked:$stop,
    MO_IndicesVectorIfRanked:$step,
    MO_Chain:$inChain,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Collective ops
//===----------------------------------------------------------------------===//

class MO_DistributedOp<string name, list<Trait> traits = []> :
  MO_TensorOp<name, [
    DeclareOpInterfaceMethods<MO_MutableOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    MO_SameVariadicOperandSize,
    MO_SameVariadicResultSize,
  ] # traits> {
  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs Variadic<MO_Tensor>:$outputs);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_AllReduceSumOp : MO_DistributedOp<"distributed.allreduce.sum", [
  AllTypesMatch<["inputs", "outputs"]>,
  MO_MultiDevice,
]> {
  let summary = "Allreduce sum across N devices.";
  let description = [{
    Sums same-shaped inputs, one per device, across devices.

    A nonzero `group_size` smaller than the input count splits devices into
    contiguous independent groups (e.g. TP groups under DP); shapes then only
    need to match within a group.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    DefaultValuedAttr<I64Attr, "0">:$group_size
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` (` ` `[` `group_size` `=` $group_size^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BundledExpandOp : Op<MODialect, "bundled.expand", [
  MO_BundledCollective,
  MO_PreservedDuringKernelLowering,
  Pure,
]> {
  let summary = "Expand a per-device tensor to all launches' tensors.";
  let description = [{
    Inside an `mo.parallel` body, expands a single-device tensor (typically a
    block argument) into one tensor per launch, making collective N-expansion
    explicit in the IR. Results match the input's shape and dtype, one per
    launch, on each launch's device (from the first input bundle).

    Example:
    ```mlir
    mo.parallel (%arg) in (%dt : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>,
                                              !mo.tensor<[3], f32, gpu:1>]>)
        -> (!mo.bundle<[...]>) {
      %peer0, %peer1 = mo.bundled.expand(%arg)
          : !mo.tensor<[3], f32, gpu:0>
         -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
      // ... use %peer0, %peer1 as inputs to a collective ...
    }
    ```
  }];
  let arguments = (ins MO_Tensor:$input);
  let results = (outs Variadic<MO_Tensor>:$results);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input) `->` `(` type($results) `)`
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_BundledAllReduceSumOp : MO_TensorOp<"bundled.allreduce.sum", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_BundledCollective,
  MO_SameVariadicOperandSize,
  MO_SingleDeviceWithPeerInputs<>,
]> {
  let summary = "Per-device allreduce sum, for use inside mo.parallel.";
  let description = [{
    Per-device allreduce sum inside `mo.parallel`. Takes N peer tensors (from
    `mo.bundled.expand`), N signal buffers captured from graph scope (chain
    guarded by the parent's `buffers(...)` clause), and a chain.

    Example:
    ```mlir
    mo.parallel (%arg) in (%dt : !mo.bundle<[...]>)
        buffers(%sig0 : ..., %sig1 : ...) chain(%ch) -> (...) {
      %peer0, %peer1 = mo.bundled.expand(%arg)
          : !mo.tensor<[3], f32, gpu:0>
         -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
      %out, %ch_out = mo.bundled.allreduce.sum(
          %peer0, %peer1, %sig0, %sig1, %ch)
          : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
             !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>,
             !mo.chain)
          -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
      mo.yield %out : !mo.tensor<[3], f32, gpu:0>
    }
    ```
  }];
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// BundledBroadcastOp
//===----------------------------------------------------------------------===//

def MO_BundledBroadcastOp : MO_TensorOp<"bundled.broadcast", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_BundledCollective,
  MO_SingleDeviceWithPeerInputs<>,
]> {
  let summary = "Per-device broadcast of the root device's tensor.";
  let description = [{
    The per-device form of `mo.distributed.broadcast`, one op per
    participating device: copies `input`, the root device's tensor, into
    this device's output. Every copy, the root's included, reads `input`,
    which the device-graph lowering routes through a stable address slot.
    `root` is the root's index in the signal buffer list, which holds one
    buffer per device.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    MO_Tensor:$input,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    DefaultValuedAttr<I64Attr, "0">:$root
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` (` ` `[` `root` `=` $root^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// BundledAllGatherOp
//===----------------------------------------------------------------------===//

def MO_BundledAllGatherOp : MO_TensorOp<"bundled.allgather", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_BundledCollective,
  MO_SameVariadicOperandSize,
  MO_SameVariadicResultSize,
  MO_SingleDeviceWithPeerInputs<>,
]> {
  let summary = "Per-device allgather over every peer's tensor.";
  let description = [{
    The per-device form of `mo.distributed.allgather`, one op per device and
    full world only: takes every peer's tensor, which the device-graph
    lowering routes through stable address slots, and one signal buffer per
    device, and produces this device's gathered pieces, one per peer in peer
    order, each with the peer's shape on this device. A grouped allgather
    routes part of a shard through a paired group's buffers, which a device
    graph cannot express, so `group_size` has no per-device form.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_ReduceScatterSumOp : MO_DistributedOp<"distributed.reducescatter.sum", [
  MO_MultiDevice,
]> {
  let summary = "ReduceScatter sum across N devices.";
  let description = [{
    Sums inputs, one per device, and gives each device a disjoint part of the
    result.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<I64Attr, "0">:$group_size
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` `` custom<ReduceScatterSumAttrs>($axis, $group_size)
    attr-dict `:` functional-type(operands, results)
  }];
}


def MO_DistributedScatterOp : MO_DistributedOp<"distributed.scatter", [
  MO_MultiDevice,
]> {
  let summary = "Scatter different chunks from root GPU to device groups.";
  let description = [{
    Sends each of ngpus inputs on device `root` (padded from dp_size distinct
    chunks) to the corresponding GPU.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    I64Attr:$root
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` `[` `root` `=` $root `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_AllGatherOp : MO_DistributedOp<"distributed.allgather", [
  MO_HasDeviceInterface,
  MO_MultiDevice,
]> {
  let summary = "AllGather across N devices (variadic).";
  let description = [{
    Concatenates inputs, one per device, along dimension 0 and replicates the
    result on every device.
  }];
  let hasVerifier = 1;
  // Folds the single-input case.
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    DefaultValuedAttr<I64Attr, "0">:$group_size
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );

  let assemblyFormat = [{
    `(` operands `)` (` ` `[` `group_size` `=` $group_size^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_EPDispatchOp : MO_DistributedOp<"distributed.ep.dispatch", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = "Multi-device Expert Parallelism BF16 dispatch.";
  let description = [{
    Dispatches tokens to expert devices across N GPUs with BF16 output. Every
    variadic operand and result has one entry per device.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$topkIds,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    Variadic<MO_Tensor>:$rowOffsets,
    Variadic<MO_Tensor>:$expertIds,
    Variadic<MO_Tensor>:$srcInfo,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_EPDispatchFP8Op : MO_DistributedOp<"distributed.ep.dispatch.fp8", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = "Multi-device Expert Parallelism FP8 dispatch.";
  let description = [{
    Like `mo.distributed.ep.dispatch`, but quantizes output to blockwise FP8.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$topkIds,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert,
    StrAttr:$dispatch_scale_granularity
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    Variadic<MO_Tensor>:$outputScales,
    Variadic<MO_Tensor>:$rowOffsets,
    Variadic<MO_Tensor>:$expertIds,
    Variadic<MO_Tensor>:$srcInfo,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert `,`
        `dispatch_scale_granularity` `=` $dispatch_scale_granularity `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

def MO_EPDispatchBlockScaledNVOp : MO_DistributedOp<"distributed.ep.dispatch.block.scaled.nv", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = "Multi-device Expert Parallelism NVFP4/MXFP4/MXFP8 dispatch.";
  let description = [{
    Like `mo.distributed.ep.dispatch`, but quantizes tokens to NVFP4, MXFP4,
    or MXFP8 and sends them via shared-memory or NVSHMEM pointers.

    `sendPtrs`, `recvPtrs`, and `recvCountPtrs` are host-side pointer tensors,
    usually identical across devices and replicated N times to satisfy the
    same-variadic-size constraint.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$topkIds,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Tensor>:$inputScales,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    Variadic<MO_Tensor>:$outputScales,
    Variadic<MO_Tensor>:$rowOffsets,
    Variadic<MO_Tensor>:$scalesOffsets,
    Variadic<MO_Tensor>:$expertIds,
    Variadic<MO_Tensor>:$srcInfo,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_EPDispatchBlockScaledNVDynGlobalScalesOp : MO_DistributedOp<
    "distributed.ep.dispatch.block.scaled.nv.dyn_global_scales", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = [{
    Multi-device Expert Parallelism NVFP4 dispatch with per-token global
    scales.
  }];
  let description = [{
    Like `mo.distributed.ep.dispatch.block.scaled.nv` in NVFP4, but quantizes
    each token against its own global scale, `2688 / rowmax`, instead of a
    caller-supplied input scale. The scale's BF16 inverse travels with the
    token and lands in `outputRowwiseScales`, one entry per row of
    `outputTokens`.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$topkIds,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    Variadic<MO_Tensor>:$outputScales,
    Variadic<MO_Tensor>:$outputRowwiseScales,
    Variadic<MO_Tensor>:$rowOffsets,
    Variadic<MO_Tensor>:$scalesOffsets,
    Variadic<MO_Tensor>:$expertIds,
    Variadic<MO_Tensor>:$srcInfo,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_EPDispatchMXFP4Op : MO_DistributedOp<"distributed.ep.dispatch.mxfp4", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = "Multi-device Expert Parallelism MXFP4 dispatch.";
  let description = [{
    Like `mo.distributed.ep.dispatch.block.scaled.nv`, but quantizes to MXFP4
    and sends via shared-memory or ROCSHMEM pointers.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$topkIds,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$mx_format
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    Variadic<MO_Tensor>:$outputScales,
    Variadic<MO_Tensor>:$rowOffsets,
    Variadic<MO_Tensor>:$expertIds,
    Variadic<MO_Tensor>:$srcInfo,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert
        (`,` `mx_format` `=` $mx_format^)? `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_EPCombineOp : MO_DistributedOp<"distributed.ep.combine", [
  MO_MultiDeviceFrom<"inputTokens">,
]> {
  let summary = "Multi-device Expert Parallelism fused combine.";
  let description = [{
    Returns expert outputs to their source devices across N GPUs and computes
    the router-weighted sum. The output supports epilogue fusion. Every
    variadic operand and result has one entry per device.
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputTokens,
    Variadic<MO_Tensor>:$srcInfo,
    Variadic<MO_Tensor>:$sendPtrs,
    Variadic<MO_Tensor>:$recvPtrs,
    Variadic<MO_Tensor>:$recvCountPtrs,
    Variadic<MO_Tensor>:$routerWeights,
    Variadic<MO_Buffer>:$atomicCounters,
    MO_Chain:$inChain,
    I64Attr:$hidden_size,
    I64Attr:$top_k,
    I64Attr:$n_experts,
    I64Attr:$max_token_per_rank,
    I64Attr:$n_gpus_per_node,
    I64Attr:$n_nodes,
    BoolAttr:$fused_shared_expert
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputTokens,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)`
    `[` `hidden_size` `=` $hidden_size `,` `top_k` `=` $top_k `,`
        `n_experts` `=` $n_experts `,`
        `max_token_per_rank` `=` $max_token_per_rank `,`
        `n_gpus_per_node` `=` $n_gpus_per_node `,` `n_nodes` `=` $n_nodes `,`
        `fused_shared_expert` `=` $fused_shared_expert `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

// Not an MO_DistributedOp: it has a single root input rather than one per
// device, so execution devices come from the outputs.
def MO_DistributedBroadcastOp : MO_Op<"distributed.broadcast", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_HasDeviceInterface,
  MO_MultiDeviceFrom<"outputs">,
  MO_TensorOpInterface,
]> {
  let summary = "Broadcast from root device to all participating devices.";
  let description = [{
    Replicates `input` from the root device to all participating devices.
    `root` is the root's index in the signal buffer list.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let arguments = (ins
    MO_Tensor:$input,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    DefaultValuedAttr<I64Attr, "0">:$root
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` (` ` `[` `root` `=` $root^ `]`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// "Custom" GPU only ops
//===----------------------------------------------------------------------===//

// TODO: Make into custom ops once custom ops can be marked CPU- or GPU-only;
// device specificity is currently tied to individual ops.

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
