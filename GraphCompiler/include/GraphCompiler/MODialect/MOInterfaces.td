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
// This defines interfaces in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD

include "Mojo/KGENDialect/KGENInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MO_TensorOpInterface : OpInterface<"MOTensorOpInterface"> {
  let description = [{
    Implemented by all MO ops; provides typed input/output accessors.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get an input tensor as a TypedValue",
      "TensorValue", "getInputTensor",
      (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TensorValue>($_op->getOperand(idx)); }]>,

    InterfaceMethod<"Get an output tensor as a TypedValue",
      "TensorValue", "getOutputTensor",
      (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TensorValue>($_op->getResult(idx)); }]>
  ];
}

def MO_ConditionallyInPlaceInterface : OpInterface<"MOConditionallyInPlaceInterface"> {
  let description = [{
    Ops that may represent an in-place computation (e.g. a custom op on an
    mo.buffer, or a load/store-fused mogg.routine). Use with
    MOMutableOpInterface.
  }];

  let methods = [
    InterfaceMethod<"Query if this op currently represents an in-place computation.",
                    "bool", "isInPlace", (ins)>,
  ];

  let cppNamespace = "::M::MO";
}

def MO_MutableOpInterface : OpInterface<"MOMutableOpInterface", []> {
  let description = [{
    Implemented by all mutable rmo/mo ops and any op that may compute in
    place. Conditionally in-place ops (e.g. mo.custom) also implement
    MOConditionallyInPlaceInterface.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get the input chain as a ChainValue",
      "llvm::SmallVector<::M::MO::ChainValue>",
      "getInChains", (ins),
      [{}],
      [{ return {$_op.getInChain()}; }]>,

    InterfaceMethod<"Get the output chain as a ChainValue",
      "llvm::SmallVector<::M::MO::ChainValue>",
      "getOutChains", (ins),
      [{}],
      [{ return {$_op.getOutChain()}; }]>,

    InterfaceMethod<"Get the input chain as a OpOperand ref",
      "llvm::SmallVector<::mlir::OpOperand *>",
      "getInChainsMutable", (ins),
      [{}],
      [{ return {&($_op.getInChainMutable())}; }]>
  ];

  let extraClassDeclaration = [{
    std::optional<::M::MO::ChainValue> getUniqueInChain() {
      auto chains = getInChains();
      if (chains.size() != 1)
        return {};
      return chains[0];
    }

    std::optional<::M::MO::ChainValue> getUniqueOutChain() {
      auto chains = getOutChains();
      if (chains.size() != 1)
        return {};
      return chains[0];
    }

    ::mlir::OpOperand *getUniqueInChainMutable() {
      auto chains = getInChainsMutable();
      if (chains.size() != 1)
        return nullptr;
      return chains[0];
    }
  }];
}

def MO_HasDeviceInterface : OpInterface<"MOHasDeviceInterface"> {
  let description = [{
    Ops with an optional `#M.device_ref` execution device, which must resolve
    to an `#M.device_spec` in the enclosing `mo.graph`'s 'device_specs'.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<
      "Returns the devices this operation executes on.",
      "llvm::SmallVector<DeviceRefAttr>", "getExecutionDevices", (ins)
    >,

    InterfaceMethod<
      "Returns the single device this operation executes on.",
      "DeviceRefAttr", "getExecutionDevice", (ins), "",
      [{
        auto devices = $_op.getExecutionDevices();
        assert(devices.size() == 1 && "Operation must execute on exactly one device");
        return devices[0];
      }]
    >,
  ];
}

class MO_TensorOpInterfaceBase<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces> {
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_TensorOpInterface];
}

def MO_ControlOpInterface : OpInterface<"MOControlOpInterface"> {
  let description = [{
    Marks control-flow ops.
  }];
  let cppNamespace = "::M::MO";
}

def MO_ParamDeclaration : MO_TensorOpInterfaceBase<"ParamDeclarationInterface",
  [MemoryEffectsOpInterface, ParamOpInterface]> {
  let description = [{
    Ops that declare shape or dimension parameters.
  }];

  let methods = [
    InterfaceMethod<
      "Return the output shape or dimension parameters declared by this op",
      "llvm::ArrayRef<KGEN::ParamDeclAttr>", "getOutputParamDecls",
      (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<
      "Set the output parameter declarations associated with this op",
      "void", "setOutputParamDecls",
      (ins "ArrayRef<KGEN::ParamDeclAttr>":$paramDecls), [{}],
      [{ assert(paramDecls.empty()); }]>,
  ];

  let extraTraitClassDeclaration = [{
    void getEffects(
      llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      if (!paramDeclItf.getOutputParamDecls().empty())
        effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    }

    void walkDeclarations(function_ref<void(KGEN::ParamDeclAttr)> walkDecl) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      llvm::for_each(paramDeclItf.getOutputParamDecls(), walkDecl);
    }

    void walkDefinitions(function_ref<void(KGEN::ParamDeclAttr,
        const KGEN::ParamDefValue &)> walkDef) {
      mlir::AttrTypeWalker walker;
      SmallVector<Attribute> refs;
      walker.addWalk([&](KGEN::ParamDeclRefAttr declRef) {
        refs.emplace_back(declRef);
      });
      Operation *op = this->getOperation();
      for (auto t : op->getOperandTypes())
        walker.walk(t);

      auto paramDeclItf = cast<ParamDeclarationInterface>(op);
      KGEN::ParamDefValue deps;
      deps.exprs.assign(refs.begin(), refs.end());
      for (KGEN::ParamDeclAttr decl : paramDeclItf.getOutputParamDecls())
        walkDef(decl, deps);
    }

    void renameDeclarations(ArrayRef<KGEN::ParamDeclAttr> decls) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      assert(decls.size() == paramDeclItf.getOutputParamDecls().size());
      paramDeclItf.setOutputParamDecls(decls);
    }
  }];
}

def MO_Staticization : OpInterface<"Staticization"> {
  let description = [{
    Ops whose output may be expressible as a parameter expression given
    staticized inputs, i.e. one of:
    - `IntegerAttr` for a integer or boolean constant, e.g., `42`.
    - `BoolAttr` for a boolean constant, e.g., `true`.
    - `KGEN::ParamDeclRefAttr` for a parameter reference, e.g., `D0`.
    - `KGEN::ParamOperatorAttr` for a parameter expression, e.g., `add(D1, 2)`.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Try to staticize this op given staticized inputs. Returns "
    "an error (fatal), a null `TypedAttr` (inputs too dynamic), or the result.",
      "ErrorOr<TypedAttr>", "tryStaticize",
      (ins "ArrayRef<TypedAttr>":$inputs,
           "ParamExprBuilder&":$b),
      [{}]>,
  ];
}

def MO_ShapeMaterialization : MO_TensorOpInterfaceBase<"ShapeMaterialization", [MO_ParamDeclaration]> {
  let description = [{
    Ops that declare parameters depending on their input shape or data, and
    may (best effort) define them in terms of new ops.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<
    "Return indices of inputs whose data (not shape) the output parameters "
    "need. TODO(GEX-4093): derivable from a "
    "trait-bound shape function's written bounds once this op migrates.",
    "SmallVector<int32_t>", "getDataDependentInputIndices",
      (ins), [{}]>,

    InterfaceMethod<
    "Try returning ops (in use-def order) defining this op's declared shape "
    "parameters, given symbolic input shapes or data `inputAttrs`.",
    "ShapeMaterializeResult", "materializeShapeDefs",
      (ins "ArrayRef<TypedAttr>":$inputAttrs),
    [{}]>,
  ];
}

def MO_MatmulLike : MO_TensorOpInterfaceBase<"MatmulLike"> {
  let description = [{
    Matmul-like ops (matmul, batchmatmul). Dynamic shapes are allowed; when
    known, the last 2 dimensions of each input are the matrix dimensions. The
    left and right matrices are A and B.
  }];

  let methods = [
    InterfaceMethod<"Get the left input tensor as a TypedValue",
      "TensorValue", "getTensorA", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TensorValue", "getTensorB", (ins), [{}],
      [{ return $_op.getInputTensor(1); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getTensorResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>
  ];

  let verify = [{ return Impl::verifyMatmulLike(cast<MatmulLike>($_op)); }];
}

def MO_SlidingWindow : MO_TensorOpInterfaceBase<"SlidingWindow"> {
  let description = [{
    Ops with sliding window semantics.
  }];

  let methods = [
    InterfaceMethod<"Get the strides as a TypedValue",
      "TensorValue", "getStrides", (ins)>,
    InterfaceMethod<"Get the dilations as a TypedValue",
      "TensorValue", "getDilations", (ins)>,
    InterfaceMethod<"Get the paddings as a TypedValue",
      "TensorValue", "getPaddings", (ins)>
  ];

  let extraClassDeclaration = [{
    LogicalResult verifyWindowParamShape(
      TensorValue $windowParam, StringRef name);
    LogicalResult verifyWindowParamSizes(ArrayRef<TensorValue> values);
  }];

  let verify = [{ return Impl::verifySlidingWindow(cast<SlidingWindow>($_op)); }];
}

def MO_ElementWiseUnary : MO_TensorOpInterfaceBase<"ElementWiseUnary"> {
  let description = [{
    Unary element-wise ops.
  }];

  let methods = [
    InterfaceMethod<"Get the input tensor as a TypedValue",
      "TensorValue", "getInput", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>,
    InterfaceMethod<"Set the input tensor to given Value",
      "void", "setInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(0, val); }]>,
  ];
}

def MO_ElementWiseBinary : MO_TensorOpInterfaceBase<"ElementWiseBinary"> {
  let description = [{
    Binary element-wise ops.
  }];

  let methods = [
    InterfaceMethod<"Get the left input tensor as a TypedValue",
      "TensorValue", "getLhsInput", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TensorValue", "getRhsInput", (ins), [{}],
      [{ return $_op.getInputTensor(1); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>,
    InterfaceMethod<"Set the left input tensor to given Value",
      "void", "setLhsInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(0, val); }]>,
    InterfaceMethod<"Set the right input tensor to given Value",
      "void", "setRhsInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(1, val); }]>,
  ];
}

def MO_Reduction : MO_TensorOpInterfaceBase<"Reduction",
    [MO_ParamDeclaration]> {
  let description = [{
    Reduction ops.
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the non-negative reduction axis, if known",
      "std::optional<int64_t>", "getReductionAxis", (ins)>,
    // TODO(GEX-3348): remove `getAxisAttribute` and `setAxisAttribute` once
    // every reduction reads its axis from the shapes.
    InterfaceMethod<"Get the `axis` attribute, or null for a reduction that "
      "reads its axis from the shapes",
      "::mlir::IntegerAttr", "getAxisAttribute", (ins)>,
    InterfaceMethod<"Set the `axis` attribute; a no-op for a reduction that "
      "reads its axis from the shapes",
      "void", "setAxisAttribute", (ins "int64_t":$axis)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyReduction(cast<Reduction>($_op)); }];
}

def MO_ScatterLike : MO_TensorOpInterfaceBase<"ScatterLike",
    [MO_ParamDeclaration,]> {
  let description = [{
    Scatter and scatter-with-reduction ops.
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the updates tensor as a TypedValue",
      "TensorValue", "getUpdates", (ins)>,
    InterfaceMethod<"Get the updates tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getUpdatesMutable", (ins)>,
    InterfaceMethod<"Get indices tensor as a TypedValue",
      "TensorValue", "getIndices", (ins)>,
    InterfaceMethod<"Get indices tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getIndicesMutable", (ins)>,
    InterfaceMethod<"Get axis tensor as a TypedValue",
      "TensorValue", "getAxis", (ins)>,
    InterfaceMethod<"Get axis tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getAxisMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyScatterLike(cast<ScatterLike>($_op)); }];
}

def MO_ScatterNdLike : MO_TensorOpInterfaceBase<"ScatterNdLike",
    [MO_ParamDeclaration]> {
  let description = [{
    ScatterND and ScatterND-with-reduction ops.
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the updates tensor as a TypedValue",
      "TensorValue", "getUpdates", (ins)>,
    InterfaceMethod<"Get the updates tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getUpdatesMutable", (ins)>,
    InterfaceMethod<"Get indices tensor as a TypedValue",
      "TensorValue", "getIndices", (ins)>,
    InterfaceMethod<"Get indices tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getIndicesMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyScatterNdLike(cast<ScatterNdLike>($_op)); }];
}

def MO_PadLike : MO_TensorOpInterfaceBase<"PadLike", [MO_ParamDeclaration]> {
  let description = [{
    Pad ops.
  }];

  let methods = [
    InterfaceMethod<"Get input as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get padding tensor as a TypedValue",
      "TensorValue", "getPaddings", (ins)>,
    InterfaceMethod<"Get result tensor type",
      "TensorType", "getType", (ins)>
  ];

  let verify = [{ return Impl::verifyPadLike(cast<PadLike>($_op)); }];
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

class MO_OpTrait<string name, list<Trait> traits = [],
    code extraOpDeclaration = [{}],
    code extraOpDefinition = [{}]>
    : NativeOpTrait<name, traits, extraOpDeclaration, extraOpDefinition> {
  let cppNamespace = "::M::MO";
}

// Marks an op that must execute on the host, with all operands and results also
// on the host
def MO_HostOnly : MO_OpTrait<"MOHostOnly">;

// Marks an op that executes on the host using tensors from any device.
def MO_ExecutesOnHost : MO_OpTrait<"MOExecutesOnHost">;

// Marks a `mo.composite.*` fused-subgraph op (see MOCompositeOps.td).
def MO_CompositeOpTrait : MO_OpTrait<"MOComposite">;

// Marks an elementwise op with heavy per-element work: a division or a
// transcendental. Kernel launch heuristics key off it.
def MO_ExpensiveElementwise : MO_OpTrait<"MOExpensiveElementwise">;

class getOperandNumbers<list<string> operandNames> {
  string ret = !interleave(!foreach(name, operandNames,
    "get" # snakeCaseToCamelCase<name>.ret
    # "Mutable().getOperandNumber()"), ", ");
}

// Marks an op that executes on a single device. hostOperandNames are on the
// host; deviceOperandNames and all results are on the device, which is
// inferred from the first device operand. If deviceOperandNames is empty, all
// non-host operands must be on the device, inferred from the first result.
class MO_SingleDeviceWithHostOperands<list<string> hostOperandNames = [],
                                      list<string> deviceOperandNames = []>
  : MO_OpTrait<"MOSingleDeviceWithHostOperands">
{
  let extraConcreteClassDeclaration = [{
    llvm::SmallVector<size_t> getHostOperandIndices() {
      return {}] # getOperandNumbers<hostOperandNames>.ret # [{};
    }
    llvm::SmallVector<size_t> getDeviceOperandIndices() {
      return {}] # getOperandNumbers<deviceOperandNames>.ret # [{};
    }
  }];
}

// Single device with all operands and results on it, inferred from the first
// result.
def MO_SingleDevice : MO_SingleDeviceWithHostOperands<>;

// Marks a multi-device op whose execution devices come from the ODS operand
// or result named deviceSourceName.
class MO_MultiDeviceFrom<string deviceSourceName>
  : MO_OpTrait<"MOMultiDevice">
{
  let extraConcreteClassDeclaration = [{
    mlir::ValueRange getDeviceSourceValues() {
      return get}] # snakeCaseToCamelCase<deviceSourceName>.ret # [{();
    }
  }];
}

def MO_MultiDevice : MO_MultiDeviceFrom<"inputs">;

// Marks an op whose execution device is its deviceRef attribute.
def MO_ExplicitDevice : MO_OpTrait<"MOExplicitDevice">;

// Single device (inferred from the first result) whose inputs may be on peer
// devices with the same label. hostOperandNames must be on the host instead.
class MO_SingleDeviceWithPeerInputs<list<string> hostOperandNames = []>
  : MO_OpTrait<"MOSingleDeviceWithPeerInputs">
{
  let extraConcreteClassDeclaration = [{
    llvm::SmallVector<size_t> getHostOperandIndices() {
      return {}] # getOperandNumbers<hostOperandNames>.ret # [{};
    }
  }];
}

def MO_SameOperandsAndResultType : TraitList<[
  SameOperandsAndResultType,
  MO_ParamDeclaration
]>;

/// On every MO and RMO op: ops consuming !mo.buffer must consume and produce
/// !mo.chain values.
def MO_BufferOpConformance : MO_OpTrait<"MOBufferOpConformance">;

/// SameOperandsAndResultShape over the MOSH::ShapeAttr of MO::Tensor.
def MO_SameOperandsAndResultShapeAttr :
    MO_OpTrait<"MOSameOperandsAndResultShapeAttr">;

def MO_SameOperandsAndResultShape: TraitList<[
  MO_SameOperandsAndResultShapeAttr
]>;

def MO_ReductionTraits: TraitList<[
  MO_Reduction,
  MO_ShapeMaterialization,
  MO_OpTrait<"ReductionShapeMaterialization">,
]>;

def MO_ScatterLikeTraits: TraitList<[
  MO_ScatterLike,
  MO_ShapeMaterialization,
  MO_OpTrait<"ScatterLikeShapeMaterialization">,
]>;

def MO_ScatterNdLikeTraits: TraitList<[
  MO_ScatterNdLike,
  MO_OpTrait<"ScatterNdLikeShapeMaterialization">,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
]>;

/// Like the builtin SameOperandsAndResultElementType, which only supports
/// mlir::ShapedType, but with MO's own element type access.
def MO_SameOperandsAndResultElementType :
    MO_OpTrait<"MOSameOperandsAndResultElementType">;

def MO_SameOperandsElementType :
    MO_OpTrait<"MOSameOperandsElementType">;

/// Results have the first operand's dtype, for shape ops like reshape and
/// squeeze whose other operands only describe the new shape.
def MO_SameFirstOperandAndResultsElementType :
    MO_OpTrait<"MOSameFirstOperandAndResultsElementType">;

def MO_PadLikeTraits: TraitList<[
  MO_PadLike,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
]>;

def MO_ConstantLike : MO_TensorOpInterfaceBase<"ConstantLike"> {
  let description = [{
    Constant ops.
  }];

  let methods = [
    InterfaceMethod<"Get result tensor type",
      "TensorType", "getType", (ins)>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getResult(0)); }]>,
  ];
}

def MO_PreservedDuringKernelLowering : OpInterface<"PreservedDuringKernelLowering"> {
  let description = [{
    An MO op that must be lowered to a kernel implementation.
  }];
  let cppNamespace = "::M::MO";
}

// Bundled collectives (e.g. `mo.bundled.expand`, `mo.bundled.allreduce.sum`)
// coordinate across launches with signal buffers: inside `mo.parallel`, from
// the parallel's `buffers(...)` clause; inside a device graph, captured from
// graph scope. `ParallelOp::verify` allows them unconditionally, and this
// interface's verifier enforces one of the two valid homes. Apply via
// `MO_BundledCollective`.
def MO_BundledCollectiveInterface
    : OpInterface<"MOBundledCollectiveInterface"> {
  let description = [{
    Marks a per-launch bundled-collective entry point, valid only inside an
    `mo.parallel` body or an `isDeviceGraph` `mo.graph` body.
  }];
  let cppNamespace = "::M::MO";

  let verify = [{ return Impl::verifyBundledCollectiveInterface($_op); }];
}

def MO_BundledCollective : TraitList<[MO_BundledCollectiveInterface]>;

def MO_ViewLike : OpInterface<"ViewLike"> {
  let description = [{
    Represents a view op.
  }];
  let cppNamespace = "::M::MO";
}

def MO_ElementWiseLike : OpInterface<"ElementWiseLike"> {
  let description = [{
    Represents a generic element-wise op.
  }];
  let cppNamespace = "::M::MO";
}

def MO_SameVariadicOperandSizeInterface : OpInterface<"SameVariadicOperandSizeInterface"> {
  let description = [{
    Exposes the builtin `SameVariadicOperandSize` trait, which C++ can't check.
  }];
  let cppNamespace = "::M::MO";
}

def MO_SameVariadicOperandSize: TraitList<[
  MO_SameVariadicOperandSizeInterface,
  SameVariadicOperandSize,
]>;

def MO_SameVariadicResultSizeInterface : OpInterface<"SameVariadicResultSizeInterface"> {
  let description = [{
    Exposes the builtin `SameVariadicResultSize` trait, which C++ can't check.
  }];
  let cppNamespace = "::M::MO";
}

def MO_SameVariadicResultSize: TraitList<[
  MO_SameVariadicResultSizeInterface,
  SameVariadicResultSize,
]>;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
