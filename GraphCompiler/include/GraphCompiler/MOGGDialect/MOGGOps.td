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
// This defines operations in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD

include "GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/MOGGDialect/MOGGEnums.td"
include "GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GraphCompiler/MOGGDialect/MOGGAttrs.td"
include "GraphCompiler/MOGGDialect/MOGGTypes.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MODialect/MOEnums.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/LITDialect/LITTypes.td"
include "Mojo/POPDialect/POPTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"


class MOGG_Op<string mnemonic, list<Trait> traits = []> :
  Op<MOGGDialect, mnemonic, traits>;

// Ops that call a kernel, a primitive or a `mogg.func`.
def MOGG_CallLike : NativeOpTrait<"CallLike"> {
  let cppNamespace = "::M::MOGG";
}

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []> :
  MOGG_Op<mnemonic, traits # [
    DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>,
  ]>;

class MOGG_KernelTerminatorOp<string mnemonic, list<Trait> traits = []> :
  MOGG_Op<mnemonic, traits # [
    DeclareOpInterfaceMethods<MOGG_KernelTerminator>,
    Terminator,
  ]>;

def DimensionType : ScalarOfDType<BoundAnd<DTypeConstraint<"*$_self == KGEN::KGENDType::si64">>, "!kgen.simd<1, si64>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::si64))">;

//===----------------------------------------------------------------------===//
// Internal helper ops
//===----------------------------------------------------------------------===//

def MOGG_ConcatOp : MOGG_LowerableOp<"concat", [
  MO_HasDeviceInterface,
  MO_SingleDevice,
  Pure,
]> {
  let summary = "Concat Operation along axis zero";
  let description = [{
    Example:

    ```mlir
    %arg0: !mo.tensor<[3, 3], f32>
    %arg1: !mo.tensor<[2, 3], f32>
    %res = mogg.concat(%arg0, %arg1) : (
      !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 3], f32>
    ) -> !mo.tensor<[5, 3], f32>
    ```

    Works for CUDA and host devices.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def MOGG_SplitOp : MOGG_LowerableOp<"split", [Pure]> {
  let summary = "Split Operation along axis zero";
  let description = [{
    `mo.split` with `axis` zero; the split sizes are implied by the result
    types.

    Example:

    ```mlir
    %input: !mo.tensor<[9, 8], f32>
    %res = mogg.split(%input) : (!mo.tensor<[9, 8], f32> ) ->
    (!mo.tensor<[2, 8], f32>, !mo.tensor<[3, 8], f32>, !mo.tensor<[4, 8], f32>)
    ```

    Currently only for the 'host' device.
    TODO: Generalize to arbitrary devices.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(`
      qualified(type($input)) `)` `->` `(` type($results) `)`
  }];
}

def MOGG_SliceOp : MOGG_Op<"slice", [
  MO_HasDeviceInterface,
  MO_SingleDeviceWithHostOperands<["start", "stop"]>,
  Pure,
]> {
  let summary = "Contiguous zero-copy slice op lowerable to mgp.tensor.slice";
  let description = [{
    Unlike mo.slice, not lowered to a kernel by MOToMOGG. FuseViews lowers it
    to a mogg.routine temporarily and raises it back if unfused.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndexScalarIfRanked:$start,
    MO_IndexScalarIfRanked:$stop
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MOGG_OutputOp : MOGG_Op<"output", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ReturnLike,
  Terminator,
]> {
  let summary = "MOGG output operation";
  let description = [{
    Specifies the output values of a region in a MOGG operation.

    ```mlir
      mogg.op foo() {
      ... mogg.output e0, % 1 : i32, f8
    } : i32, f8
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


//===----------------------------------------------------------------------===//
// MOGG abstract compiled components representation
//===----------------------------------------------------------------------===//

def MOGG_CallOp : MOGG_Op<"call", [
  MOGG_CallLike,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  MOGG_FuncCallOpInterface,
]> {
  let summary = "Calls into an outlined stub";
  let description = [{}];
  let arguments = (ins
    FlatSymbolRefAttr:$name,
    Variadic<AnyType>:$inputs,
    OptionalAttr<I64Attr>:$affinity_device_id
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $name ($inputs^ `:` type($inputs))? (`->` type($outputs)^)?
    (`affinity_device_id` `=` $affinity_device_id^)? attr-dict
  }];

  let extraClassDeclaration = [{
    FuncOp getCallee();
  }];
}

def MOGG_CallAsyncOp : MOGG_Op<"call.async", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  MOGG_FuncCallOpInterface,
]> {
  let summary = "Calls into an outlined stub asynchronously";
  let description = [{
    Dispatches via Mojo `create_task`. `affinity_device_id` pins the task to a
    CPU core with NUMA affinity for that GPU. It is a plain integer because
    kernel-layer device refs are label-only.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$name,
    Variadic<AnyType>:$inputs,
    OptionalAttr<I64Attr>:$affinity_device_id
  );
  let results = (outs MOGG_TaskHandle:$task_handle);
  let assemblyFormat = [{
    $name (`(` $inputs^ `:` type($inputs) `)`)?
    (`affinity_device_id` `=` $affinity_device_id^)? attr-dict
  }];

  let extraClassDeclaration = [{
    FuncOp getCallee();
  }];
}

def MOGG_CallBuildGraphOp : MOGG_Op<"call.build_graph", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Record a kernel's `build_graph` function into a device-graph builder";
  let description = [{
    The device-graph counterpart of `mogg.call.execute`: invokes `build_graph`
    against a builder instead of `execute` against a context, recording a node
    rather than running the kernel. Produces no result; the enclosing
    `mgp.device_graph.dependencies` op orders nodes. Not a kernel terminator,
    and not wrapped in `mogg.with_trace`.

    `$hasEpilogueFusion` mirrors `mogg.call.opaque`: it is copied from the
    dispatch this call records and set the kernel's `has_epilogue_fusion`
    parameter, so a fused body reaches `build_graph` the same way it reaches
    `execute`.

    Example:
    ```mlir
    mogg.call.build_graph["linalg_kernels.__init__.Matmul"]
        inputs(%in, %wt : !mo.tensor<...>, !mo.tensor<...>)
        outputs(%out : !mo.tensor<...>)
        builder(%builder)

    // Lowered to
    Matmul.build_graph[...](out, in, wt, builder)
    ```
  }];

  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    UnitAttr:$isThrows,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$inputs,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$outputs,
    MGP_DeviceGraphBuilderType:$graphBuilder,
    OptionalAttr<DictionaryAttr>:$kernel_param,
    DefaultValuedAttr<BoolAttr, "false">:$hasEpilogueFusion
  );

  let assemblyFormat = [{
    `[` $kernelSymName `]` `` custom<OptionalKernelParams>($kernel_param)
    (`inputs` `(` $inputs^ `:` qualified(type($inputs)) `)`)?
    (`outputs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
    `builder` `(` $graphBuilder `)`
    (`isThrows` $isThrows^)?
    `moOpName` `=` $moOpName
    (`hasEpilogueFusion` `=` $hasEpilogueFusion^)?
    attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_AsyncRoutineOp : MOGG_Op<"routine.async", [
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
  DeclInterface,
  IsolatedFromAbove,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  SingleBlockImplicitTerminator<"MOGG::OutputOp">,
]> {
  let summary = "An abstract function representing a compiled body of code";
  let description = [{
    Takes and returns AsyncValues and runs its body to completion (always
    blocking). A generic form of mogg.routine; the first result is a mef.chain.

    Sequence fusion (concatenating code) fuses a parent region into a child:
    `canFuseIntoOther` allows this region as parent, `canBeFusedInto` as
    child. `canDuplicate` lets a region with several children be copied into
    each of them.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    MO_ParamDecls:$outputParamDecls,
    M_InOutSignatureAttr:$inOutSignature,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Immediate">:$dispatch,
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canFuseIntoOther,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canBeFusedInto,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canDuplicate,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isStructured
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<ImmediateDispatch>($dispatch)
    custom<Kernel>($operands, type($operands), type($results), $body,
                   $outputParamDecls)
    custom<FusionInfo>($fusionInfo)
    (`canBeFusedInto` `=` $canBeFusedInto^)?
    (`canDuplicate` `=` $canDuplicate^)?
    (`canFuseIntoOther` `=` $canFuseIntoOther^)?
    (`constantResources` `=` $constantResources^)?
    `inOutSignature` `=` $inOutSignature
    (`isStructured` `=` $isStructured^)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    /// Required by DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }

    Block &getBodyBlock() { return getBody().front(); }

    MOGG::OutputOp getBodyTerminator();
  }];
}

def MOGG_GetDimOp : MOGG_Op<"get_dim", [Pure]> {
  let summary = "Returns the runtime representation of a dimension";

  let arguments = (ins Builtin_StringAttr:$dimension);
  let results = (outs DimensionType:$result);

  let assemblyFormat = [{
    custom<Dim>($dimension) attr-dict
  }];
}

def MOGG_ToMGPFromOpaqueOp : MOGG_Op<"to_mgp_from_opaque", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the opaque pointer from the kgen value";
  let description = [{
    A no-op that keeps IR types correct while converting a KGEN opaque value
    (e.g. mo.opaque for graph outputs) to !mgp.opaque. Folds away when the
    types match.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasFolder = 1;
}

def MOGG_ScratchPlaceholdersOp : MOGG_Op<"scratch_placeholders"> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "Represents scratch buffer tensors that are not returned from the kernel";
  let description = [{
    Like `mogg.output_placeholders`, but the tensors are not returned from the
    kernel.

    Example:
    ```mlir
      %scratch = mogg.scratch_placeholders : !mo.tensor<[2, 3], f32>
    ```
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_CompiledLoadOp : MOGG_LowerableOp<"compiled.load"> {
  let summary = "Loads pairs of (symbol, compiled object) into the runtime "
                "for a given device.";
  let description = [{
    Registers each symbol with its compiled-object blob in the runtime, for
    use by ops like `CompiledCallOp`. The object format and registration
    depend on the op's device.
  }];

  let arguments = (ins
    ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value,
    DefaultValuedAttr<BoolAttr, "false">:$debuginfo
  );
}


def MOGG_CompiledCallOp : MOGG_Op<"compiled.call", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
]> {
  let summary = "Executes a compiled kernel on a given device.";
  let description = [{
    Looks up the compiled object (loaded by `CompiledLoadOp`) by name
    and calls it with the lowered operands.

    `usedParams` are the shape parameters the kernel uses; they are passed at
    runtime to reconstruct tensor shapes. `outputParamDecls` defines parameter
    values, which MGP translates into parameter system calls. `dispatch` says
    how the runtime schedules the call (see `MOGG_DispatchKind`).

    The kernel reads and writes memory its operands only point to, such as
    the buffer an init-block call fills, so the op is never pure: a call
    whose results are unused is not dead, and two identical calls are not
    redundant.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    FlatSymbolRefAttr:$calleeSymbol,
    ArrayAttr:$usedParams,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Immediate">:$dispatch,
    ArrayAttr:$constantResources
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $calleeSymbol `` custom<OutputParamDecls>($outputParamDecls)
    `` `(` $operands `)`
    custom<ImmediateDispatch>($dispatch)
    custom<FusionInfo>($fusionInfo)
    `constantResources` `=` $constantResources `,`
    `usedParams` `=` $usedParams
    attr-dict `:` functional-type($operands, $results)
  }];
}

def MOGG_FuncOp : MOGG_Op<"func", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  FunctionOpInterface,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"MOGG::OutputOp">,
  SymbolName,
  SymbolVisibility,
  Symbol,
]> {
  let summary = "Op that outlines a kernel";
  let description = [{
    A kernel that will be compiled to Mojo text.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    MOGG_FusionInfoAttr:$fusionInfo,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Immediate">:$dispatch,
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
    MOGG_ABIKindAttr:$abi,
    // Execution-device label for each kernel's `target` parameter. A string
    // rather than `#M.device_ref` because only the label matters.
    StrAttr:$device,
    OptionalAttr<ArrayAttr>:$trait_stub_buffer_capture,
    OptionalAttr<ArrayAttr>:$trait_stub_arg_bounds,
    // Needed by -symbol-dce.
    DefaultValuedAttr<StrAttr, "\"private\"">:$sym_visibility
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    OutputOp getBodyTerminator();
    Region *getCallableRegion() { return &getBodyRegion(); }
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr attrs){}
    void setResAttrsAttr(mlir::ArrayAttr attrs){}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }

  private:
    mlir::LogicalResult emitMojoMGPABI(EmitMojoState &state);
    mlir::LogicalResult emitMojoMOGGABI(EmitMojoState &state);
  }];
}

def MOGG_TensorInitOp : MOGG_Op<"tensor.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Represents the construction of a tensor";
  let description = [{
    Lowers to a ManagedTensorSlice constructor call from a data pointer and
    dimensions.
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$buffer,
    Variadic<SI64Scalar>:$dims,
    BoolAttr:$mut,
    MO_IOKind:$io
  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);

  let assemblyFormat = [{
    `(` $buffer (`,` $dims^)? `)` `io` `=` custom<IOKind>($io) `,` `mut` `=` $mut
    attr-dict `:` qualified(type($buffer)) `->` qualified(type($tensor))
  }];
}

def MOGG_TensorArgInitOp : MOGG_Op<"tensor_arg.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  AttrSizedOperandSegments,
  Pure,
]> {
  let summary = "Construct a trait-lane tensor argument from its contract";
  let description = [{
    Like `mogg.tensor.__init__`, but carries the `#mogg.tensor_arg` contract
    for an argument satisfying a kernel trait bound, instead of `{io, mut}`.
    `role`/`access` are read from the contract, not the result type.

    `$device_context` is for the GPU emit path and is not yet populated.
    `contract.fused` is currently always false and `contract.access` `none`.
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$buffer,
    Variadic<SI64Scalar>:$dims,
    Optional<MOGG_Context>:$device_context,
    MOGG_TensorArgContractAttr:$contract
  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);

  let assemblyFormat = [{
    `(` $buffer (`,` $dims^)? `)` (`context` `(` $device_context^ `)`)?
    `contract` `=` $contract
    attr-dict `:` qualified(type($buffer)) `->` qualified(type($tensor))
  }];
}

def MOGG_AsyncReadyOp : MOGG_Op<"async.ready", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let results = (outs ChainType:$output);

  let summary = "Represents a chain output placeholder";
  let description = [{
    Resolves chain dependencies for data-dependent operations and lets fusion
    elide chains when lowering to a compiled region.
  }];

  let assemblyFormat = [{
    attr-dict
  }];
}

def MOGG_AsyncJoinOp : MOGG_Op<"async.join", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let arguments = (ins Variadic<MOGG_TaskHandle>:$task_handles);

  let summary = "Waits for async task handles and raises on error";
  let description = [{
    Waits for all tasks, then raises the first error, if any. Has no results;
    use `mogg.async.ready` for the output chain.

    Example:
    ```mlir
    mogg.async.join(%t0, %t1)
    ```
  }];

  let assemblyFormat = [{
    `(` $task_handles `)` attr-dict
  }];
}

def MOGG_AsyncUnpackOp : MOGG_Op<"async.unpack", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
  TypesMatchWith<"input is the async wrapper of the result",
                 /*lhs=*/"output", /*rhs=*/"input",
                 "::M::MOGG::AsyncType::get($_self)">,
]> {
  let arguments = (ins MOGG_Async:$input);
  let results = (outs AnyType:$output);
  let summary = "Unpacks an async value and returns it as sync";
  let description = [{
    Projects `!mogg.async<T>` (values outside a `mogg.routine.async`)
    to `T`. Inverse of `mogg.async.pack`. Only `T` is printed.
  }];

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($output)
  }];

  let hasFolder = 1;
}

def MOGG_AsyncPackOp : MOGG_Op<"async.pack", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
  TypesMatchWith<"result is the async wrapper of the input",
                 /*lhs=*/"input", /*rhs=*/"output",
                 "::M::MOGG::AsyncType::get($_self)">,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs MOGG_Async:$output);
  let summary = "Packs an sync value and returns it as async";
  let description = [{
    Inverse of `mogg.async.unpack`. Only `T` is printed.
  }];

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input)
  }];

  let hasFolder = 1;
}

def MOGG_EmptyBufferOp : MOGG_Op<"empty_buffer", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Represents an empty buffer (actually of size 1) allocated on the stack";
  let description = [{
    Backs empty tensors passed to some kernels after fusion.
  }];

  let results = (outs MGP_DeviceBufferType:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_YieldOp : MOGG_Op<"yield", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ReturnLike,
  Terminator,
]> {
  let summary = "MOGG yield operation";
  let description = [{
    Yields the results of a mogg.try / except, the value a
    `mogg.tensor.create.prologue` body returns to the reading kernel, the
    value a compute `mogg.tensor.create.epilogue` body returns for the kernel
    to store, or an element's contribution to a `mogg.rowwise.reduce`.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def MOGG_WithTraceOp : MOGG_Op<"with_trace", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Wraps a kernel terminator in a Trace context manager";
  let description = [{
    Models `with Trace[...]` around the KernelTerminator of a MOGG ABI stub.
    Always throwing because Trace.__enter__ raises.
  }];
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict (`:` type($outputs)^)?";
}

def MOGG_ProfileScopeBeginOp : MOGG_Op<"profile_scope.begin", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Opens a trace range for a Graph.profile_scope() span";
  let description = [{
    Lowers to `var t = Trace[...](name, color=...); t.start()`.

    A region-less sibling of `mogg.with_trace`: values defined inside a region
    would be invisible to later ops, and sequence-fused bodies have SSA values
    live across scope boundaries.

    `color` is set only when every covered op agrees on one (see
    `resolveProfileScopeColor`). Always throwing because `Trace.start()`
    raises.
  }];
  let arguments = (ins StrAttr:$name, OptionalAttr<StrAttr>:$color);
  let results = (outs MOGG_ProfileScopeRange:$range);
  let assemblyFormat = [{
    $name (`color` `=` $color^)? attr-dict
  }];
}

def MOGG_ProfileScopeEndOp : MOGG_Op<"profile_scope.end", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Closes the trace range opened by a `mogg.profile_scope.begin`";
  let description = [{
    Lowers to `t.end()`. Always throwing because `Trace.end()` raises.
  }];
  let arguments = (ins MOGG_ProfileScopeRange:$range);
  let assemblyFormat = [{
    $range attr-dict
  }];
}

def MOGG_RaiseOp : MOGG_Op<"raise", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  Terminator,
]> {
  let summary = "Re-raises a kernel error with full context";
  let description = [{
    Terminates the except block of a MOGG ABI stub, adding the stub name,
    fusion info, and Python traceback to the error.
  }];
  let arguments = (ins MOGG_Error:$error);
  let assemblyFormat = "$error attr-dict";
}

def MOGG_TryOp : MOGG_Op<"try", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "Op that represent a try / except statement";
  let description = [{
    The except region takes the raised `!mogg.error` as its only argument,
    printed as `except(%err)`.
  }];
  let regions = (region SizedRegion<1>:$tryRegion, SizedRegion<1>:$exceptRegion);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $tryRegion custom<ExceptRegion>($exceptRegion)
    attr-dict `:` type($outputs)
  }];

  let hasCanonicalizer = 1;
}

def MOGG_AsyncErrorOp : MOGG_Op<"async.error", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let arguments = (ins MOGG_Error:$error);
  let results = (outs AnyType:$output);

  let summary = "Represents setting an error on a kernel output";
  let description = [{
    Sets an error on a chain or non-tensor output in a raising kernel's except
    block.

    ```
    mogg.try {
     ...
    } except(%err) {
      %chain = mogg.async.error %err : !mef.chain
      %dim0 = mogg.async.error %err : !si64
      mogg.yield %chain, %dim0
    }
    ```

    Lowers to `mogg_async_error_value`, returning an error `AnyAsyncValueRef`.
  }];

  let assemblyFormat = [{
    $error attr-dict `:` type(results)
  }];
}

def MOGG_ShapeResultGetItemOp : MOGG_Op<"shape_result.__getitem__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Gets a single dimension from a shape function result";
  let description = [{
    Example:
    ```
    %dim0 = mogg.shape_result.__getitem__(%shape, 0) : !mogg.shape_func_result<4>
    %dim1 = mogg.shape_result.__getitem__(%shape, 1) : !mogg.shape_func_result<4>
    ```
  }];

  let arguments = (ins
    MOGG_ShapeFuncResult:$shape,
    Builtin_IntegerAttr:$index
  );
  let results = (outs SI64Scalar:$result);

  let assemblyFormat = [{
    `(` $shape `,` $index `)` attr-dict `:` qualified(type($shape))
  }];

  let hasVerifier = 1;
}

def MOGG_ShapeInitOp : MOGG_Op<"shape.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Construct a mojo shape object";
  let description = [{
    Builds an IndexList from the static shape attribute and its dynamic
    dimensions.

    Example:
    ```
      %shape = mogg.shape.__init__() <shape = #mosh<ape[1, 2, 3, 4]> : !mosh.ape>
      %dyn_shape = mogg.shape.__init__(%d0_val) <shape = #mosh<ape[2, D0, 3]> : !mosh.ape>
    ```
  }];

  let arguments = (ins
    Variadic<SI64Scalar>:$dynamic_dims,
    MOSH_ShapeAttr:$shape
  );
  let results = (outs MOGG_Shape:$result);

  let assemblyFormat = [{
    `(` $dynamic_dims `)` `<` `shape` `=` qualified($shape) `>` attr-dict
  }];
}

def MOGG_ChainDummyOp : MOGG_Op<"chain.dummy", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Returns a chain to feed MRP primitives";
  let description = [{
    MGP primitives lowered into a kernel still consume a chain, which sequence
    fusion may turn into an intermediate result, so they are fed this no-op.
    TODO(GEX-2764): drop the chain from the signature when lowering instead.
  }];

  let arguments = (ins);
  let results = (outs ChainType:$chain);

  let assemblyFormat = [{
    `(` `)` attr-dict
  }];
}

def MOGG_ChainSinkOp : MOGG_Op<"chain.sink", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Consumes a chain and does nothing with it";
  let description = [{
    Keeps MGP primitives that lower to kernels from being DCE'd when their
    result has no consumer.
  }];

  let arguments = (ins ChainType:$chain);
  let results = (outs);

  let assemblyFormat = [{
    `(` $chain `)` attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}


//===----------------------------------------------------------------------===//
// KernelDeclOp
//===----------------------------------------------------------------------===//

def MOGG_KernelDeclOp
    : MOGG_Op<"kernel.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares a Mojo kernel's signatures inline in the module";
  let description = [{
    Holds the signatures of a Mojo kernel struct registered under an MO op
    name. Several decls may share a `registration` (e.g. per-device
    implementations). A user-defined decl (`is_user_defined`) overrides
    built-ins with the same registration. The shape function lives in a
    separate `mogg.shape_func.decl`.

    `build_graph` has the same tensor signature as `execute` but takes a
    `DeviceGraphBuilder` in place of the `DeviceContext`.

    `execute_device` and `execute_configuration` make a declarative kernel:
    `execute_device` is the device code, taking the operands but no context,
    and `execute_configuration` is a host function over the same operands
    returning its launch configuration: any value with `get_grid_dim()` and
    `get_block_dim()`, as NeoDriver's `GridBlockExecutionConfig` has. The graph
    compiler writes the launch itself. A kernel has either `execute` or
    `execute_device`.

    `reduction` marks a `ReduceOp` monoid (see `#mogg.kernel_reduction`).

    `supports_unfused_execute` records a second `execute` overload that takes
    every fused output as a plain `Tensor` output, which the graph compiler
    calls when nothing is fused into any of them. The overload is not stored:
    `ImportPackages` checks it against `execute` and drops it, and
    `ConformCallOperands` derives the plain output contracts from `execute`.
    Grandfathered kernels only; see `kUnfusedExecuteRegistrations`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$registration,
    M_DeviceInfoAttr:$registration_device_info,
    OptionalAttr<MOGG_KernelFunctionAttr>:$execute,
    OptionalAttr<MOGG_KernelFunctionAttr>:$elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$tile_elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$build_graph,
    OptionalAttr<MOGG_KernelFunctionAttr>:$execute_device,
    OptionalAttr<MOGG_KernelFunctionAttr>:$execute_configuration,
    OptionalAttr<MOGG_KernelReductionAttr>:$reduction,
    ArrayAttr:$struct_params,
    BoolAttr:$is_user_defined,
    UnitAttr:$supports_unfused_execute,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) `` $sym_name
    `registration` `=` $registration
    `device_info` `=` $registration_device_info
    (`execute` `=` $execute^)?
    (`elementwise` `=` $elementwise^)?
    (`tile_elementwise` `=` $tile_elementwise^)?
    (`build_graph` `=` $build_graph^)?
    (`execute_device` `=` $execute_device^)?
    (`execute_configuration` `=` $execute_configuration^)?
    (`reduction` `=` $reduction^)?
    `struct_params` `=` $struct_params
    `is_user_defined` `=` $is_user_defined
    (`supports_unfused_execute` $supports_unfused_execute^)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    /// The name the struct's location carries among the op's locations; each
    /// function's carries `stringifyKernelFunctionKind`'s.
    static constexpr ::llvm::StringLiteral kStructLocationName = "struct";

    /// Returns the Mojo location of the kernel's struct, falling back to the
    /// op's location.
    ::mlir::Location getMojoStructLocation();

    /// Returns the Mojo location of the kernel's `kind` function, falling
    /// back to the struct's. A shape function lives in its own decl, so
    /// `Shape` gets the struct's.
    ::mlir::Location getMojoFunctionLocation(KernelFunctionKind kind);

    /// Returns the name argument `position` of the `kind` function has among
    /// the op's locations, e.g. "execute.arg2".
    static std::string getMojoArgLocationName(KernelFunctionKind kind,
                                              uint64_t position);

    /// Returns the Mojo location of argument `position` of the `kind`
    /// function, falling back to `getMojoFunctionLocation(kind)`.
    ::mlir::Location getMojoArgLocation(KernelFunctionKind kind,
                                        uint64_t position);
  }];
}

def MOGG_PrimitiveDeclOp
    : MOGG_Op<"primitive.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares a primitive Mojo function inline in the module";
  let description = [{
    Records the signature of a standalone `@register_internal` Mojo function
    (e.g. the `_foreach` driver). `sym_name` is its fully-qualified name and
    `registration` its intrinsic string (e.g. `"mogg.call.foreach"`). A
    primitive must have exactly one registration, since the symbol table
    rejects a duplicate `sym_name`. `result_params` is not populated.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    StrAttr:$registration,
    MOGG_KernelFunctionAttr:$function
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) `` $sym_name
    `registration` `=` $registration `function` `=` $function attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ShapeFunctionDeclOp
//===----------------------------------------------------------------------===//

def MOGG_ShapeFunctionDeclOp
    : MOGG_Op<"shape_func.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares the shape function for a kernel op registration";
  let description = [{
    At most one per registration (e.g. `"mo.avg_pool"`); import errors on
    duplicates. For a struct `shape` method, `sym_name` is
    `"<struct_sym>.shape"`. `result_params` is always populated, since shape
    functions return stdlib `IndexList`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    StrAttr:$registration,
    MOGG_KernelFunctionAttr:$function
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) `` $sym_name
    `registration` `=` $registration `function` `=` $function attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen ops
//
// `MAPToMOGG` lowers the `map` dialect into the ops below. Values are opaque
// handles; shape/dtype/mut/io live on the creating op.
//===----------------------------------------------------------------------===//

def MOGG_RoutineOp : MOGG_LowerableOp<"routine", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  NoTerminator,
]> {
  let summary =
      "A codegen kernel: `!mo.tensor` inputs/outputs outside, `!mogg.pointer` "
      "inside.";
  let description = [{
    Lowered from `map.iter.opaque`. Inputs are `!mo.tensor` (or `!mo.buffer`
    for an in-place kernel `FuseMutableLoads` fused a buffer into) and outputs
    `!mo.tensor`. The entry-block arguments are one `!mogg.pointer` per input,
    one per output, then the dynamic `!mogg.dim`s, then a `!mogg.context`.
    There is no terminator: the output pointer arguments are the outputs.

    The optional `chain` sequences an in-place kernel and has no block
    argument; the output chain is the last of `outputs`. `device` gives the
    execution device so outlining need not fall back to the host. `fusionInfo`
    is absent only for hand-built kernels (e.g. in tests).
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer, MO_Opaque]>>:$inputs,
    Optional<MO_Chain>:$chain,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    MO_ParamDecls:$outputParamDecls,
    // Without these, a kernel whose only effect is a mutable store looks dead.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Deferred">:$dispatch
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Chain, MO_Opaque,
                                          MOGG_ShapeFuncResult]>>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the last entry of `outputs` if it is a chain, else null.
    ::mlir::Value getOutputChain();

    /// Lowers to a `mogg.routine.async` and registers its outputs;
    /// returns null on error. `mo.parallel` calls it per launch to collect
    /// regions it fuses into one async region.
    AsyncRoutineOp
    lowerToAsync(::M::MGP::MOGraphLoweringState &loweringState);
  }];
}

def MOGG_ShapeCreateOp : MOGG_Op<"shape.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Build a codegen shape from a static shape attr + dynamic dims.";
  let description = [{
    `dynamic_dims` are in shape order, from the enclosing `mogg.routine` block
    arguments.
  }];
  let arguments = (ins Variadic<MOGG_Dim>:$dynamic_dims, MOSH_ShapeAttr:$shape);
  let results = (outs MOGG_Shape:$result);
  let assemblyFormat = [{
    `(` $dynamic_dims `)` `<` `shape` `=` qualified($shape) `>`
    attr-dict (`:` qualified(type($dynamic_dims))^)?
  }];
}

def MOGG_TensorCreateOp : MOGG_Op<"tensor.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor from a pointer, shape and static metadata.";
  let description = [{
    Reconstructs a tensor inside a kernel body. `$device` (cpu vs gpu) lets
    emit pick the preferred alignment without consulting the kernel stub.
  }];
  let arguments = (ins
    MOGG_Pointer:$ptr,
    MOGG_Shape:$shape,
    MOSH_ShapeAttr:$shapeAttr,
    StrAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io,
    // The source `!mo.tensor` was a stable address slot: emit builds a
    // `StableTensor` over the slot instead of a pointer-backed slice.
    DefaultValuedAttr<BoolAttr, "false">:$stable
  );
  // `!mo.buffer` reconstructs as `!mogg.buffer`, so `getKernelOperandKind()`
  // resolves `Buffer` vs `Tensor` from the type alone.
  let results = (outs AnyTypeOf<[MOGG_Tensor, MOGG_Buffer]>:$tensor);
  let assemblyFormat = [{
    `(` $ptr `,` $shape `)`
    `<` `device` `=` $device `,` `elementType` `=` $elementType `,`
        `io` `=` custom<IOKind>($io) `,` `mut` `=` $mut `,`
        `shapeAttr` `=` qualified($shapeAttr)
        (`,` `stable` `=` $stable^)? `>`
    attr-dict `:` qualified(type($tensor))
  }];
}

def MOGG_TensorCreateEpilogueOp : MOGG_Op<"tensor.create.epilogue", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  AttrSizedOperandSegments,
  NoTerminator,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_FusedEpilogueInterface>,
]> {
  let summary = "A fused-output tensor bound to its own epilogue body.";
  let description = [{
    A `!mogg.tensor` that writes through its own epilogue body, lowered from
    `map.iter.epilogue`. The body's entry-block arguments are the
    `!mogg.simd` value and its index, then the captures. `$device` (cpu vs
    gpu) lets emit pick the preferred alignment.

    A store body ends with `mogg.tensor.store` into the store-target capture.
    A compute body ends with `mogg.yield` of the value to store, leaving the
    store to the kernel; the store-target capture stays so emit has a slice to
    bind onto.

    `$computeOutputArg` records whether the kernel declares this output as a
    `_FusedComputeOutputTensor`, choosing `_bind_to_fused_compute_output` over
    `_bind_to_fused_output`. It is independent of the body shape: a
    dtype-changing epilogue or `enable-compute-lambdas=false` keeps a store
    body against a compute-capable argument.
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    Variadic<MOGG_TensorLike>:$captures,
    Variadic<MOGG_Shape>:$shapeCaptures,
    MOGG_Pointer:$outputPtr,
    StrAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io,
    DefaultValuedAttr<BoolAttr, "false">:$computeOutputArg,
    // Present iff the kernel declares this output against a trait bound, so
    // emit wraps the functor in the trait tensor instead of a
    // `ManagedTensorSlice`. Set at MAP lowering, where the call edge is known.
    OptionalAttr<MOGG_TensorArgContractAttr>:$contract
  );
  let results = (outs MOGG_Tensor:$tensor);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MOGG_TileCreateEpilogueOp : MOGG_Op<"tile.create.epilogue", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  NoTerminator,
  DeclareOpInterfaceMethods<MOGG_FusedEpilogueInterface>,
]> {
  let summary = "A tile fused-output tensor bound to its own epilogue body.";
  let description = [{
    The tile counterpart of `mogg.tensor.create.epilogue`: a `!mogg.tensor`
    whose epilogue runs once per tile the kernel produces, not once per
    element. The body's leading arguments are the kernel's tile and the tile
    index; one argument per capture follows, the first being the store
    target. A load in the body reads a whole tile of a capture at the index.

    `$lane` follows the kernel's declared output kind. A compute lane ends in
    `mogg.yield` of the transformed tile, which the kernel stores; a store
    lane ends in a store into the store-target capture and owns the write.
    Unlike the SIMD epilogue the two never decouple: a tile epilogue cannot
    change dtype or transform its index, so there is no fallback lane.

    Example:
    ```mlir
    %epi = mogg.tile.create.epilogue %shape (%val : !mogg.simd<f32>, %idx : !map.index<2>)
             capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor}
             ptr %ptr lane = compute, device = "gpu", elementType = f32,
             io = Output, mut = true {
      %zv = mogg.tensor.load %cz[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%val, %zv)
               moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.yield %sum : !mogg.simd<f32>
    }
    ```
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    Variadic<MOGG_TensorLike>:$captures,
    MOGG_Pointer:$outputPtr,
    MOGG_TileFusionLaneAttr:$lane,
    StrAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io
  );
  let results = (outs MOGG_Tensor:$tensor);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MOGG_TensorGetPtrOp : MOGG_Op<"tensor.get_ptr", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Extract the raw `!mogg.pointer` backing a codegen tensor.";
  let description = [{
    Gives a `mogg.tensor.create.epilogue` the pointer of its store target (or
    of the pre-cast accumulation buffer for a mixed-precision accumulating
    kernel), so the fused output is built over real storage that an
    accumulating kernel can accumulate through.
  }];
  // A buffer when the epilogue stores into a fused `mo.mutable.store`.
  let arguments = (ins MOGG_TensorLike:$tensor);
  let results = (outs MOGG_Pointer:$ptr);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` qualified(type($tensor))
  }];
}

def MOGG_TensorCreatePrologueOp : MOGG_Op<"tensor.create.prologue", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  AttrSizedOperandSegments,
  SingleBlock,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "A fused-input tensor bound to its own prologue body.";
  let description = [{
    Load-side counterpart of `mogg.tensor.create.epilogue`, lowered from
    `map.iter.prologue`. The body takes the index followed by one rebuilt tile
    per capture and yields the value the kernel reads.
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    Variadic<MOGG_TensorLike>:$captures,
    // Runtime shapes for a dynamic `mogg.index.reshape` in the body; they
    // become trailing `IndexList` fields on the fusion functor.
    Variadic<MOGG_Shape>:$shapeCaptures,
    StrAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io,
    // Present iff the kernel declares this input against a trait bound, so emit
    // wraps the functor in `_MoggInputLoadFusionTensor`.
    OptionalAttr<MOGG_TensorArgContractAttr>:$contract
  );
  let results = (outs MOGG_Tensor:$tensor);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MOGG_TensorStoreOp : MOGG_Op<"tensor.store", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Store a SIMD value into a codegen tensor at an index.";
  let arguments = (ins MOGG_TensorLike:$tensor, AnyType:$index, MOGG_Simd:$value);
  let assemblyFormat = [{
    $tensor `[` $index `]` `=` $value attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($index)) `,` qualified(type($value))
  }];
}

def MOGG_CallOpaqueOp : MOGG_Op<"call.opaque", [
  MOGG_CallLike,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Call a registered opaque kernel's execute function.";
  let description = [{
    Invokes `$callee`'s `execute`. `$device_contexts` passes the device
    contexts explicitly: one for a single-device kernel, or one per device for
    a `DeviceContextArray`, which `ConformCallOperands` packs into a tuple.
    `$hasEpilogueFusion` spares emit from inspecting output producers.

    A kernel with `execute_device` is launched rather than called: emit calls
    its `execute_configuration`, then enqueues `execute_device` on the device
    context itself.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$inputs,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$outputs,
    Variadic<AnyTypeOf<[MOGG_Context, MOGG_Tuple]>>:$device_contexts,
    OptionalAttr<DictionaryAttr>:$kernel_param,
    DefaultValuedAttr<BoolAttr, "false">:$hasEpilogueFusion,
    // The source op's `MO_SameVariadicOperandSize` trait, lost when it becomes
    // a call, so `ConformCallOperands` can split operands into equal-length
    // variadic groups (e.g. a collective's N inputs and N signal buffers).
    DefaultValuedAttr<BoolAttr, "false">:$hasSameVariadicOperandSize
  );
  // An `!mo.opaque` result is allocated inside the kernel body, so it has no
  // destination `$output`; `mogg.routine` forwards it with `mogg.output`.
  let results = (outs Optional<MO_Opaque>:$opaqueResult);
  let assemblyFormat = [{
    $callee `` custom<OptionalKernelParams>($kernel_param)
    (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
    (`outputs` `(` $outputs^ `:` type($outputs) `)`)?
    (`context` `(` $device_contexts^ `:` type($device_contexts) `)`)?
    (`hasEpilogueFusion` `=` $hasEpilogueFusion^)?
    (`hasSameVariadicOperandSize` `=` $hasSameVariadicOperandSize^)?
    attr-dict (`->` type($opaqueResult)^)?
  }];
}

def MOGG_ScalarCreateOp : MOGG_Op<"scalar.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Pack a single-element tensor operand into a scalar operand.";
  let description = [{
    Inserted by `ConformCallOperands` for an operand the kernel expects as a
    scalar.
  }];
  let arguments = (ins MOGG_Tensor:$tensor);
  let results = (outs MOGG_Scalar:$scalar);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict
  }];
}

def MOGG_TraitCreateOp : MOGG_Op<"trait.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Re-present a codegen tensor as its conforming trait argument.";
  let description = [{
    Inserted by `ConformCallOperands` for an operand the callee declares
    against a trait bound (`Tensor` / `TileTensorable`), which
    a `ManagedTensorSlice` does not satisfy. Emit wraps it in
    `_MoggDenseTensorArg` (execute) or `_MoggShapeTensorArg` (shape,
    `role = none`) per `$contract`. The result keeps the operand's
    tensor/buffer kind, so the call's emit is unchanged. Works over any
    producer, including non-row-major views. `$device` is as for
    `mogg.tensor.create`.
  }];
  let arguments = (ins
    MOGG_TensorLike:$tensor,
    StrAttr:$device,
    MOGG_TensorArgContractAttr:$contract
  );
  let results = (outs MOGG_TensorLike:$result);
  let assemblyFormat = [{
    `(` $tensor `)` `contract` `=` $contract `,` `device` `=` $device
    attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TupleCreateOp : MOGG_Op<"tuple.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Pack a run of variadic tensor operands into a tuple operand.";
  let description = [{
    New-path counterpart of `mogg.as_tuple`: `ConformCallOperands`
    inserts this to pack the consecutive `!mogg.tensor` operands that satisfy a
    single variadic kernel argument into one `!mogg.tuple`. For a `TensorTuple`
    (`$variadic_abi = tensor_tuple`) the operands are the elements' trait
    arguments. It also packs the per-device `!mogg.context` operands of a
    distributed kernel into a tuple-of-contexts, which emit renders as a
    `DeviceContextArray`.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Context]>>:$elements,
    DefaultValuedAttr<MOGG_VariadicABIAttr,
        "::M::MOGG::VariadicABI::Plain">:$variadic_abi);
  let results = (outs MOGG_Tuple:$tuple);
  let assemblyFormat = [{
    `(` $elements `)` (`variadic_abi` `=` $variadic_abi^)?
    attr-dict (`:` qualified(type($elements))^)?
  }];
}

def MOGG_RebindOp : MOGG_Op<"rebind", [Pure,
    DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "A cast between an unpacked async handle and a codegen handle.";
  let description = [{
    E.g. `!mgp.buffer` -> `!mogg.pointer`, which emits Mojo extracting the raw
    data pointer from the `OwnedByteBuffer`; `!mogg.context` / `!mogg.dim`
    casts emit nothing.

    TODO: remove once `mogg.async.unpack` folds directly to the typed handle.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `to` type($output)
  }];
}

def MOGG_BufferConstantOp : MOGG_Op<"buffer.constant", [Pure]> {
  let summary = "A constant buffer, as a raw codegen pointer.";
  let description = [{
    A `mo.constant` that `FuseConstantIterOpaqueInputs` folded into a
    `map.iter.opaque` body, used like an input pointer. Carries no alignment,
    since the minimum CPU alignment is known only in `MOGGToMGP`, which
    rewrites it to `mgp.buffer.constant` plus a `mogg.rebind`.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs MOGG_Pointer:$pointer);
  let assemblyFormat = [{
    $value attr-dict
  }];
}

def MOGG_TensorLoadOp : MOGG_Op<"tensor.load", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Load a SIMD value from a codegen tensor at an index.";
  let description = [{
    Not `Pure`, so it is not reordered across a store to the same tensor.
  }];
  let arguments = (ins MOGG_TensorLike:$tensor, AnyType:$index);
  let results = (outs MOGG_Simd:$result);
  let assemblyFormat = [{
    $tensor `[` $index `]` attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($index)) `->` qualified(type($result))
  }];
}

def MOGG_TileLoadOp : MOGG_Op<"tile.load", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Load a tile of a codegen tensor at a tile coordinate.";
  let description = [{
    Reads the whole tile of `$tensor` at `$coord`, the tile counterpart of
    `mogg.tensor.load`. Emission stages each load into its own on-chip tile,
    so the op is not `Pure`: merging two loads would merge their staging
    tiles. Where `$tensor`'s element type is known from its creating op, it
    must match the result tile's.

    Example:
    ```mlir
    %t = mogg.tile.load %x[%coord] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    ```
  }];
  let arguments = (ins MOGG_TensorLike:$tensor, MOGG_TileCoord:$coord);
  let results = (outs MOGG_Tile:$result);
  let assemblyFormat = [{
    $tensor `[` $coord `]` attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($coord)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MOGG_TileStoreOp : MOGG_Op<"tile.store", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Store a tile into a codegen tensor at a tile coordinate.";
  let description = [{
    Writes `$value` as the tile of `$tensor` at `$coord`, the tile counterpart
    of `mogg.tensor.store`. Where `$tensor`'s element type is known from its
    creating op, it must match the stored tile's.

    Example:
    ```mlir
    mogg.tile.store %out[%coord] = %t : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
    ```
  }];
  let arguments = (ins MOGG_TensorLike:$tensor, MOGG_TileCoord:$coord,
                       MOGG_Tile:$value);
  let assemblyFormat = [{
    $tensor `[` $coord `]` `=` $value attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($coord)) `,` qualified(type($value))
  }];
  let hasVerifier = 1;
}

def MOGG_ConstantScalarOp : MOGG_Op<"constant.scalar", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "A scalar constant inlined into a fusion body as a literal.";
  let description = [{
    In a SIMD body the result is a `!mogg.simd`, as `mogg.tensor.load`
    produces, so it can replace a load; it is emitted as a literal splatted to
    the closure's SIMD width. The one builtin scalar result is the literal a
    tile call takes next to its tiles. Not `ConstantLike`, since that hoisting
    is wanted only on `map.constant.scalar`.

    Example:
    ```mlir
    %cst = mogg.constant.scalar 2.000000e+00 : f32 : !mogg.simd<f32>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTypeOf<[MOGG_Simd, AnyFloat, AnyInteger]>:$result);
  let assemblyFormat = [{
    $value attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MOGG_CallForeachOp : MOGG_Op<"call.foreach", [
  MOGG_CallLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  NoTerminator,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Drive a parallel foreach loop over a fused elementwise body.";
  let description = [{
    Runs the generic foreach driver `$callee` over `$shape`. The body does the
    per-index load-compute-store; its arguments are one per `$capturing`
    tensor, then the index. Lowered from a `map.iter.opaque` wrapping a
    `map.iter.elementwise.body`.

    Captures may differ in shape from `$shape` and each other (e.g.
    broadcasts), so there is no shape verifier. A capture is a `!mogg.buffer`
    where in-place fusion folded in a mutable buffer.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    MOGG_Shape:$shape,
    MOGG_Context:$context,
    Variadic<MOGG_TensorLike>:$capturing,
    StrAttr:$device
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def MOGG_CallForeachTileOp : MOGG_Op<"call.foreach.tile", [
  MOGG_CallLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  NoTerminator,
]> {
  let summary = "Drive a tile foreach loop over a fused elementwise body.";
  let description = [{
    The tile counterpart of `mogg.call.foreach`: `$callee`, the tile foreach
    driver, runs the body once per tile of `$shape`. The body's arguments are
    one per `$capturing` tensor, then the tile index; it loads whole tiles of
    its input captures at the index, computes, and stores exactly once into
    its output capture.

    Example:
    ```mlir
    mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile
        shape(%s) context(%ctx) (%idx : !map.index<2>)
        capturing {%co = %out : !mogg.tensor, %cx = %x : !mogg.tensor}
        device = "gpu" {
      %xv = mogg.tensor.load %cx[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %co[%idx] = %xv : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    MOGG_Shape:$shape,
    MOGG_Context:$context,
    Variadic<MOGG_TensorLike>:$capturing,
    StrAttr:$device
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MOGG_CallViewCopyOp : MOGG_Op<"call.view_copy", [
  MOGG_CallLike,
]> {
  let summary = "Copy a view into a tensor through the registered view-copy primitive.";
  let description = [{
    Materializes `$input`, a view (broadcast, transpose, slice, reshape) that
    the device cannot read in place, into `$output` by calling `$callee`, the
    `view_copy_impl` primitive. The primitive reads the view through the
    source slice's own strides, so it needs no loop body of its own, unlike
    the load/store body a `mogg.call.foreach` would run for the same copy.

    Example:
    ```mlir
    mogg.call.view_copy @builtin_primitives.primitives.view_copy_impl
        (%out, %bcast : !mogg.tensor, !mogg.tensor) context(%ctx) device = "gpu"
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    MOGG_TensorLike:$output,
    MOGG_TensorLike:$input,
    MOGG_Context:$context,
    StrAttr:$device
  );
  let assemblyFormat = [{
    $callee `(` $output `,` $input `:` type($output) `,` type($input) `)`
    `context` `(` $context `)` `device` `=` $device attr-dict
  }];
}

def MOGG_CallRowwiseOp : MOGG_Op<"call.rowwise", [
  MOGG_CallLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  NoTerminator,
]> {
  let summary = "Run a fused reduction body once per row.";
  let description = [{
    Runs its body once per row of `$shape` along `$axis`. The body holds the
    row's folds (`mogg.rowwise.reduce`), row-level ops and stores. Its
    arguments are one per `$capturing` tensor, then the row index, which has
    the `$axis` coordinate pinned to 0. Lowered from a `map.iter.opaque`
    wrapping a `map.iter.rowwise.body`.

    `mogg-legalize-rowwise-kernels` sets the rest, which the Row API needs:

    - `$primary`: the capture the folds walk along `$axis`.
    - `$dtype`: `$primary`'s element type.
    - `$accum`: the type the folds' contributions are computed in.
    - `$is_cached`: whether more than one fold reads the row.
    - `$cols`: the static size of `$axis`.
    - `$num_phases`: 1 when the body is reduce-shaped.
    - `$dtype_size`: `$primary`'s byte size when the body may split the
      row, else 0.
    - `$computationally_expensive`: whether the folds' per-element work
      reaches the pass's cost threshold.
    - `$simd_monoid`: the fold monoid that picks the SIMD width, the one
      with the largest state.

    Example:
    ```mlir
    mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>)
        capturing {%x = %X : !mogg.tensor, %out = %OUT : !mogg.tensor}
        device = "cpu" {
      %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum
          (%i : !map.index<2>) -> !mogg.simd<f32> {
        %v = mogg.tensor.load %x[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
        mogg.yield %v : !mogg.simd<f32>
      }
      mogg.tensor.store %out[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
    ```
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    I64Attr:$axis,
    MOGG_Context:$context,
    Variadic<MOGG_TensorLike>:$capturing,
    StrAttr:$device,
    OptionalAttr<I64Attr>:$primary,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<TypeAttr>:$accum,
    UnitAttr:$is_cached,
    OptionalAttr<I64Attr>:$cols,
    OptionalAttr<I64Attr>:$num_phases,
    OptionalAttr<I64Attr>:$dtype_size,
    UnitAttr:$computationally_expensive,
    OptionalAttr<FlatSymbolRefAttr>:$simd_monoid
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def MOGG_RowwisePrimaryOp : MOGG_Op<"rowwise.primary", [
  HasParent<"MOGG::RowwiseReduceOp">,
  Pure,
]> {
  let summary = "The primary input's value at the current element.";
  let description = [{
    Stands for a load of the `primary` capture at the fold's own index. The
    row is loaded once and handed to each fold.
  }];
  let results = (outs MOGG_Simd:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_RowwiseEmitOp : MOGG_Op<"rowwise.emit", [
  HasParent<"MOGG::CallRowwiseOp">,
  NoTerminator,
  SingleBlock,
]> {
  let summary = "The per-row stores of a rowwise kernel.";
  let description = [{
    Holds the stores at the row index. Only one participant of a row writes.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def MOGG_RowwiseReduceOp : MOGG_Op<"rowwise.reduce", [
  HasParent<"MOGG::CallRowwiseOp">,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  SingleBlock,
]> {
  let summary = "Fold the current row with a monoid.";
  let description = [{
    Folds every element of the enclosing row with `$monoid`. The body runs
    once per element, at the index in its argument, and yields the
    element's contribution. The results are the monoid's output fields.

    `$params` binds the monoid's dtype parameters and `$fields` names the
    output field behind each result.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$monoid,
    OptionalAttr<DictionaryAttr>:$params,
    OptionalAttr<StrArrayAttr>:$fields
  );
  let results = (outs Variadic<MOGG_Simd>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MOGG_CallShapeOp : MOGG_Op<"call.shape", [
  MOGG_CallLike,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Call a registered shape function, returning its shape by value.";
  let description = [{
    `$kernel_param` carries `output_rank`, the rank of `$result`.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    UnitAttr:$isThrows,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Scalar, MOGG_Tuple]>>:$operands,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs MOGG_ShapeFuncResult:$result);
  let assemblyFormat = [{
    $callee `` custom<OptionalKernelParams>($kernel_param) ``
    `(` $operands `:` type($operands) `)` `->` type($result)
    (`isThrows` $isThrows^)? attr-dict
  }];
}

def MOGG_CallElementwiseOp : MOGG_Op<"call.elementwise", [
  MOGG_CallLike,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure,
]> {
  let summary = "Call the registered `::elementwise` function for a particular kernel";
  let description = [{
    Calls the `::elementwise` function of the kernel named by `$callee`,
    taking the iteration index as an explicit operand.

    Example:
    ```mlir
    mogg.call.elementwise @"mo.add"(...)
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    StrAttr:$moOpName,
    Variadic<AnyType>:$inputs,
    Optional<AnyType>:$index,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $callee `` custom<OptionalKernelParams>($kernel_param) ``
    `(` $inputs (`,` $index^)? `)` `moOpName` `=` $moOpName
    attr-dict `:` type($inputs) (`,` type($index)^)? `->` type($outputs)
  }];

  // `KernelCallInterface::getKernelSymNameAttr` must hand back the leaf
  // `StringAttr` (the same interned attr the kernel decl's `sym_name` carries,
  // so symbol-liveness set membership still works); the callee is stored as a
  // `FlatSymbolRefAttr` here rather than a bare string the way `call.opaque`
  // does.
  let extraClassDeclaration = [{
    mlir::StringAttr getKernelSymNameAttr() {
      return getCalleeAttr().getLeafReference();
    }
  }];
}

def MOGG_TensorCreateSliceOp : MOGG_Op<"tensor.create.slice", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a strided slice view of another codegen tensor.";
  let description = [{
    A zero-copy view indexed by the consuming loop's index.
    `$outShape`/`$staticStart`/`$staticStep` hold the static parts (`?` for
    runtime-only dimensions).
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    DenseI64ArrayAttr:$staticStart,
    DenseI64ArrayAttr:$staticStep,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Tensor:$starts,
    MOGG_Tensor:$stops,
    MOGG_Tensor:$steps
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $starts `,` $stops `,` $steps `)`
    `<` `elementType` `=` $elementType `,` `mut` `=` $mut `,`
        `outShape` `=` qualified($outShape) `,`
        `staticStart` `=` qualified($staticStart) `,`
        `staticStep` `=` qualified($staticStep) `>`
    attr-dict `:` qualified(type($input))
  }];
}

def MOGG_TensorCreateBroadcastOp : MOGG_Op<"tensor.create.broadcast", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a broadcast view of another codegen tensor.";
  let description = [{
    A zero-copy view with numpy broadcasting semantics, indexed by the
    consuming loop's index.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `)`
    `<` `elementType` `=` $elementType `,` `mut` `=` $mut `,`
        `outShape` `=` qualified($outShape) `>`
    attr-dict `:` qualified(type($input))
  }];
}

def MOGG_TensorCreateTransposeOp : MOGG_Op<"tensor.create.transpose", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a transpose view of another codegen tensor.";
  let description = [{
    A zero-copy view indexed by the consuming loop's index.
    `$staticPermutations[i]` is the `$input` dimension that becomes output
    dimension `i` (`?` if known only at runtime, then read from
    `$permutations`).
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    DenseI64ArrayAttr:$staticPermutations,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Tensor:$permutations
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $permutations `)`
    `<` `elementType` `=` $elementType `,` `mut` `=` $mut `,`
        `outShape` `=` qualified($outShape) `,`
        `staticPermutations` `=` qualified($staticPermutations) `>`
    attr-dict `:` qualified(type($input))
  }];
}

def MOGG_TensorCreateReshapeOp : MOGG_Op<"tensor.create.reshape", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a reshape view of another codegen tensor.";
  let description = [{
    A zero-copy row-major view indexed by the consuming loop's index.
    `$outShape` is the static shape (`dyn` where unknown) and `$shape` the
    runtime one, built with `mogg.shape.create` so identical shapes CSE.
    Unlike broadcast/transpose, the result's stride comes from the output
    shape alone, so a wrong runtime dim corrupts every element past the first
    row.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Shape:$shape
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $shape `)`
    `<` `elementType` `=` $elementType `,` `mut` `=` $mut `,`
        `outShape` `=` qualified($outShape) `>`
    attr-dict `:` qualified(type($input))
  }];
}

def MOGG_IndexReshapeOp : MOGG_Op<"index.reshape", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Reindex an N-D index into an M-D index preserving row-major offset.";
  let description = [{
    The per-index counterpart of `mogg.tensor.create.reshape`, for a reshape
    fused into an epilogue's store. Not hoisted, since it depends on the loop
    index. `$dynamicShapes` supplies runtime shapes for any dynamic dims and
    is empty when both shapes are static.

    Example:
    ```mlir
    %j = mogg.index.reshape(%i) <fromShape = #mosh<ape[8, 8]> : !mosh.ape, toShape = #mosh<ape[64]> : !mosh.ape> : (!map.index<2>) -> !map.index<1>
    ```
  }];
  let arguments = (ins
    AnyType:$index,
    MOSH_ShapeAttr:$fromShape,
    MOSH_ShapeAttr:$toShape,
    Variadic<MOGG_Shape>:$dynamicShapes
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` $index (`,` $dynamicShapes^)? `)`
    `<` `fromShape` `=` qualified($fromShape) `,`
        `toShape` `=` qualified($toShape) `>`
    attr-dict `:` qualified(type($index)) `->` qualified(type($result))
  }];
}

def MOGG_IndexSliceSelectOp : MOGG_Op<"index.slice.select", [
  SingleBlock,
  NoTerminator,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Run a body at a slice's output index, where the slice selects an index.";
  let description = [{
    Lowering of `map.index.slice.select`, in place inside an epilogue functor
    body like a store-side `mogg.index.reshape`: the body runs only where the
    slice selects `$index`, with its one block argument bound to the matching
    index in the slice's output. `$inShape`/`$outShape` are the slice's static
    shapes (`dyn` where unknown) and `$staticStart`/`$staticStep` its bounds.
    `$dynamicShapes` is the runtime [in-shape, out-shape] pair when either
    shape has a dynamic dim, and empty otherwise.

    Evaluated once per stored vector at the vector's first index, so the
    innermost dim, which the vector runs along, must be unsliced.

    Example:
    ```mlir
    mogg.index.slice.select(%i) <inShape = #mosh<ape[2, 9, 16]> : !mosh.ape, outShape = #mosh<ape[2, 8, 16]> : !mosh.ape, staticStart = array<i64: 0, 1, 0>, staticStep = array<i64: 1, 1, 1>> : !map.index<3> {
    ^bb0(%j: !map.index<3>):
      mogg.tensor.store %window[%j] = %val : !mogg.tensor, !map.index<3>, !mogg.simd<f32>
    }
    ```
  }];
  let arguments = (ins
    AnyType:$index,
    MOSH_ShapeAttr:$inShape,
    MOSH_ShapeAttr:$outShape,
    DenseI64ArrayAttr:$staticStart,
    DenseI64ArrayAttr:$staticStep,
    Variadic<MOGG_Shape>:$dynamicShapes
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $index (`,` $dynamicShapes^)? `)`
    `<` `inShape` `=` qualified($inShape) `,`
        `outShape` `=` qualified($outShape) `,`
        `staticStart` `=` qualified($staticStart) `,`
        `staticStep` `=` qualified($staticStep) `>`
    attr-dict `:` qualified(type($index))
    $body
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
