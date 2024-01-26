//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGTensorAccessor.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEMOGGFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {

// The decorators we will look for on the generator to identify it as a MO
// kernel.
constexpr StringLiteral registerDecorator =
    "$stdlib::$utils::$_annotations::mogg_register";
// TODO(#27757): Temporary as transition to Mojo async/await.
constexpr StringLiteral willBecomeAsyncDecorator =
    "$stdlib::$utils::$_annotations::mogg_will_become_async";
constexpr StringLiteral registerOverrideDecorator =
    "$stdlib::$utils::$_annotations::mogg_register_override";

constexpr StringLiteral tensorAllocDecorator =
    "$stdlib::$utils::$_annotations::mogg_tensor_allocator";
constexpr StringLiteral tensorCopyConstructDecorator =
    "$stdlib::$utils::$_annotations::mogg_tensor_copy_constructor";
constexpr StringLiteral tensorDeconstructDecorator =
    "$stdlib::$utils::$_annotations::mogg_tensor_deconstructor";

constexpr StringLiteral elementwiseHook =
    "$stdlib::$utils::$_annotations::mogg_elementwise_hook";
constexpr StringLiteral tensorEnableFusion =
    "$stdlib::$utils::$_annotations::mogg_enable_fusion";
constexpr StringLiteral tensorInputFusionHook =
    "$stdlib::$utils::$_annotations::mogg_input_fusion_hook";
constexpr StringLiteral tensorOutputFusionHook =
    "$stdlib::$utils::$_annotations::mogg_output_fusion_hook";

template <typename LambdaToApply>
SmallVector<TypedAttr> forEachDecorator(GeneratorOp userKernel,
                                        LambdaToApply lambda) {
  SmallVector<TypedAttr> decoratorsToCopy;
  for (TypedAttr decorator : userKernel.getDecorators()) {
    // Keep track of the non mogg decorators to preserve them on the user
    // kernel.
    decoratorsToCopy.push_back(decorator);

    // Decorators are expected to the the apply of a symbol.
    auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator);
    if (!apply)
      continue;

    // The first operand is expected to be the symbol we are applying.
    auto sym = dyn_cast<KGEN::SymbolConstantAttr>(apply.getOperand(0));
    if (!sym)
      continue;

    StringRef decoratorName = sym.getSymbol().getLeafReference().strref();
    lambda(decorator, decoratorName, decoratorsToCopy);
  }
  return decoratorsToCopy;
}

bool isTensor(KGEN::DeclRefType maybeTensor) {
  // Look at the top level symbol name, it is structured like
  // Folder::File::ClassName.
  ArrayRef<FlatSymbolRefAttr> attr =
      maybeTensor.getSymbol().getNestedReferences();
  if (attr.size() == 0)
    return false;

  if (maybeTensor.getSymbol().getRootReference() != "$MOGGTensor")
    return false;

  StringRef className = attr[attr.size() - 1].getValue();
  if (className == "Tensor")
    return true;
  return false;
}

std::optional<LIT::LITSignatureType> getSourceSig(GeneratorOp gen) {
  std::optional<PreservedAttr> sig = gen.getSourceSignature();
  if (!sig.has_value())
    return std::nullopt;
  auto typeAttr = dyn_cast<TypeAttr>(sig.value().getValue());
  if (!typeAttr)
    return std::nullopt;
  auto litSig = dyn_cast<LIT::LITSignatureType>(typeAttr.getValue());
  if (!litSig)
    return std::nullopt;
  return litSig;
}

// Returns true if there is at least one recognizable tensor on the signature.
bool hasAtLeastOneTensor(GeneratorOp generator) {
  std::optional<LIT::LITSignatureType> litSig = getSourceSig(generator);
  if (!litSig.has_value())
    return false;

  for (Type metadata : litSig->getValues().getInputs()) {
    // Tensors are expected to be passed as references.
    auto asLitRef = dyn_cast<LIT::RefType>(metadata);
    if (!asLitRef)
      continue;

    auto asDeclRef = dyn_cast<KGEN::DeclRefType>(asLitRef.getElementType());
    if (!asDeclRef)
      continue;
    if (isTensor(asDeclRef))
      return true;
  }

  return false;
}

// Given a mojo function pull the tensor parameter information off of it. I.E
// which parameter corresponds to which parameter in a given input.
std::optional<MOGG::MOGGTensorParamAccessor>
getTensorRepFromFunctionInput(GeneratorOp generator, size_t index) {
  std::optional<PreservedAttr> sig = generator.getSourceSignature();
  if (!sig.has_value())
    return std::nullopt;
  auto typeAttr = dyn_cast<TypeAttr>(sig.value().getValue());
  auto litSig = dyn_cast<LIT::LITSignatureType>(typeAttr.getValue());

  Type metadata = litSig.getValues().getInputs()[index];

  // Tensors are expected to be passed as references.
  auto asLitRef = dyn_cast<LIT::RefType>(metadata);
  if (!asLitRef)
    return std::nullopt;

  auto asDeclRef = dyn_cast<KGEN::DeclRefType>(asLitRef.getElementType());
  if (!asDeclRef)
    return std::nullopt;
  if (!isTensor(asDeclRef))
    return std::nullopt;

  MOGG::MOGGTensorParamAccessor tensor;

  for (auto [paramIdx, param] : llvm::enumerate(asDeclRef.getParamValues())) {
    tensor.assignParam(param, paramIdx);
  }

  return tensor;
}

class SliceMOGGFuncsPass
    : public M::KGEN::MOGGPreElab::impl::SliceMOGGFuncsBase<
          SliceMOGGFuncsPass> {
private:
  struct AnnotatedKernel {
    /// Every mogg kernel should have a registration hook mapping it onto an op.
    TypedAttr moggRegister;

    /// If true, indicates the kernel will be implemented by an 'async' Mojo
    /// function.
    bool isAsync = false;

    /// When cloning the kernel we want to preserve the decorators unrelated to
    /// mogg.
    SmallVector<TypedAttr> nonMOGGDecorators;
  };

  /// Rewrite a given call to reflect a change in the parameters being passed.
  /// Which parameters are controlled by the caller of this function through the
  /// given lambda.
  void rewriteCallWithNewParams(
      CallOp call, SymbolTable symTab,
      std::function<void(const MOGG::MOGGTensorParamAccessor &,
                         SmallVector<TypedAttr> &)>
          updateParams) {
    KGEN::SymbolConstantAttr symbol = call.getCallee();
    FlatSymbolRefAttr flatSym = cast<FlatSymbolRefAttr>(symbol.getSymbol());
    auto calledFunc =
        cast<KGEN::GeneratorOp>(symTab.lookup(flatSym.getValue()));

    SmallVector<TypedAttr> newParams;
    for (TypedAttr param : symbol.getParamValues())
      newParams.push_back(param);

    // Update the parameters using the caller provided heuristic.
    for (auto [idx, value] : llvm::enumerate(call->getOperands())) {
      std::optional<MOGG::MOGGTensorParamAccessor> callRep =
          getTensorRepFromFunctionInput(calledFunc, idx);
      if (callRep.has_value())
        updateParams(*callRep, newParams);
    }

    // Now we have the list of parameters which need to be updated we
    // can rewrite the call to reflect the new lambda.
    auto newSig = calledFunc.getSignature().getSpecializedSignature(
        newParams, [&]() -> mlir::InFlightDiagnostic {
          return calledFunc->emitError(
              "INTERNAL COMPILER ERROR: Parameter specialization "
              "failed");
        });
    if (!newSig)
      signalPassFailure();

    // Point the call to the new rebinding.
    call.setCalleeAttr(
        KGEN::SymbolConstantAttr::get(flatSym, newParams, newSig));
  }

  std::optional<AnnotatedKernel> checkForMOGGAttrs(GeneratorOp userFunc) {
    AnnotatedKernel metadata;

    // Look for the mogg attributes on the kernels.
    auto lambda = [&](TypedAttr decorator, StringRef decoratorName,
                      SmallVector<TypedAttr> &attrsToCopy) {
      if (decoratorName.starts_with(registerDecorator) ||
          decoratorName.starts_with(registerOverrideDecorator)) {
        metadata.moggRegister = decorator;
        // Drop the mogg decorator
        attrsToCopy.pop_back();
      } else if (decoratorName.starts_with(willBecomeAsyncDecorator)) {
        // TODO(#27757): Temporary while transition to Mojo async/await.
        // Eventually this will be implied by the generator op's signature.
        metadata.isAsync = true;
        // Drop the mogg decorator
        attrsToCopy.pop_back();
      }
    };

    // Capture the decorators unrelated to mogg so they can be preserved.
    metadata.nonMOGGDecorators = forEachDecorator(userFunc, lambda);

    // This is not a mogg kernel if it doesn't have a register.
    if (!metadata.moggRegister)
      return std::nullopt;
    return metadata;
  }

  // We have a special mojo hook which show us what the canonical lambda
  // looks like and a call which tells us the resulting type with the lambda
  // applied.
  struct LambdaTemplate {
    LambdaTemplate() = default;

    // Scan the hook for the properties that we know exist.
    LambdaTemplate(GeneratorOp hook) : templateOp(hook) {
      for (auto region : hook.getOps<KGEN::ParamDeclareRegionOp>())
        canonicalLambda = region;
      for (auto call : hook.getOps<KGEN::CallOp>())
        callUsingLambda = call;
    }
    // The op we are pulling this info from.
    KGEN::GeneratorOp templateOp;

    // This the the template lambda we will clone as the input or output lambda.
    KGEN::ParamDeclareRegionOp canonicalLambda;

    // This call shows us how the lambda needs to be bound.
    KGEN::CallOp callUsingLambda;
  };

  MLIRContext *ctx;

  // The reference input and output lambdas we should use for materializing the
  // input/output fusion.
  LambdaTemplate inLambdaTemplate, outLambdaTemplate;

  // Each tensor carries a parameter which when set will turn off or on
  // refcounting. It defaults to refcounting so the user code remains legal.
  // Since we own the tensors force them all to false and disable refcounting /
  // memory deallocation in the kernel.
  void markAllTensorsAsOwned(GeneratorOp gen, SymbolTable &symTab) {
    auto boolType = DTypeConstantAttr::get(ctx, DType::kBool);
    auto boolFalse = POP::SIMDAttr::get({false, KGENDType::kBool},
                                        POP::SIMDType::get(1, boolType));
    gen.walk([&](CallOp call) {
      auto paramUpdate = [&](const MOGG::MOGGTensorParamAccessor &tensor,
                             SmallVector<TypedAttr> &newParams) {
        if (KGEN::ParamIndexRefAttr param = tensor.ownedMemoryAsRef())
          newParams[param.getIndex()] = boolFalse;
      };

      rewriteCallWithNewParams(call, symTab, paramUpdate);
    });
  }

  /// Instantiate all input/output lambdas which have NOT been toggled on to a
  /// None value. This will cause the tensor internally to fallback on its non
  /// lambda fusion path for load / store.
  void instantiateNoneParamLambdas(GeneratorOp gen) {
    Block *computeBlock = gen.getBody();

    int num_params = 0;
    // For now assume all variant parameters are referring to a lambda. Each
    // variant parameter will be initalized as a None type to represent no
    // fusion.
    for (auto param : gen.getInputParams()) {
      if (auto asVariant = dyn_cast<KGEN::VariantType>(param.getType())) {
        OpBuilder b{computeBlock, computeBlock->begin()};

        auto lambdaNoneTy = KGEN::VariantAttr::get(
            b.getIntegerAttr(b.getI1Type(), 0), 1, asVariant);

        std::string newalias = "_none_lambda_" + std::to_string(num_params++);
        b.create<ParamDeclareOp>(gen.getLoc(),
                                 ParamDeclAttr::get(newalias, param.getType()),
                                 lambdaNoneTy);

        ParamDeclRefAttr newRef =
            ParamDeclRefAttr::get(newalias, param.getType());
        ParamDeclRefAttr oldRef =
            ParamDeclRefAttr::get(param.getName(), param.getType());
        mlir::AttrTypeReplacer walker;
        walker.addReplacement(
            [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
              if (attr == oldRef)
                return newRef;
              return std::nullopt;
            });

        gen.walk([&](Operation *op) {
          if (op == gen) {
            walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                /*replaceLocs=*/false,
                                                /*replaceTypes=*/true);
          }
        });
      }
    }
  }

  void attachMetadataToGenerator(GeneratorOp gen,
                                 ArrayRef<std::string> inputLambdaNames,
                                 ArrayRef<std::string> outputLambdaNames,
                                 CallOp elementwiseOp, GeneratorOp shapeFunc,
                                 bool isView = false) {
    // The new attributes on the generator.
    SmallVector<NamedAttribute> newAttrs;

    // Add all the old attributes.
    for (NamedAttribute attr : gen->getAttrs())
      newAttrs.push_back(attr);

    OpBuilder b{ctx};

    // Mark this as a sliced function so MOGG lowering can identify it.
    newAttrs.push_back(
        NamedAttribute{b.getStringAttr("_sliced"), b.getUnitAttr()});

    if (isView) {
      newAttrs.push_back(
          NamedAttribute{b.getStringAttr("_view_op"), b.getUnitAttr()});
    }

    // Attach the lambda metadata.
    SmallVector<StringRef> inNames;
    for (const std::string &name : inputLambdaNames)
      inNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr("_in_lambdas"),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames;
    for (const std::string &name : outputLambdaNames)
      outNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr("_out_lambdas"),
                                      b.getStrArrayAttr(outNames)});

    if (elementwiseOp) {
      // Last parameter is known to be the lambda...
      auto elemwiseLambda =
          elementwiseOp
              .getParamValues()[elementwiseOp.getParamValues().size() - 1];
      auto asParam = dyn_cast<ParamDeclRefAttr>(elemwiseLambda);
      newAttrs.push_back(NamedAttribute{b.getStringAttr("_elementwise_lambda"),
                                        asParam.getName()});
    }

    // Tell the graph compiler that this function has an accompanying shape
    // function.
    if (shapeFunc) {
      newAttrs.push_back(NamedAttribute{b.getStringAttr("_has_shape_func"),
                                        shapeFunc.getSymNameAttr()});
    }

    gen->setAttrs(newAttrs);
  }

  // By checking which tensors have called the `enableFusion` function we can
  // use this information to enable fusion for those which have opted in.
  // Enabling fusion involves materializing a call to the input/output lambda
  // within the body of the function and replacing all previous parameter uses
  // with that value.
  void enableFusion(GeneratorOp gen, Value outputTensor,
                    SmallVector<std::string> &inputLambdaNames,
                    SmallVector<std::string> &outputLambdaNames,
                    SmallVector<KGEN::CallOp> &enableFusionFuncs,
                    SymbolTable &symTab) {
    Block *computeBlock = gen.getBody();

    for (KGEN::CallOp enableFusionFunc : enableFusionFuncs) {
      std::string newLambdaName;
      Value tensorFusionEnabledOn = enableFusionFunc.getOperand(0);

      bool isInput = true;
      LambdaTemplate *lambda;

      if (tensorFusionEnabledOn == gen.getBody()->getArgument(0)) {
        newLambdaName = "output_0_fn";
        outputLambdaNames[0] = newLambdaName;
        lambda = &outLambdaTemplate;
        isInput = false;
      } else {
        lambda = &inLambdaTemplate;

        // We are dealing with an input.
        for (auto [index, value] :
             llvm::enumerate(gen.getBody()->getArguments())) {
          if (value == tensorFusionEnabledOn) {
            // -1 to account for the first "input" being an output.
            newLambdaName = "input_" + std::to_string(index - 1) + "_fn";
            inputLambdaNames[index - 1] = newLambdaName;
            break;
          }
        }
      }

      // Instead of referring to the `self` argument of the wrapper function
      // which contains the canonical lambda we remap onto the argument of
      // this function which invoked the enable fusion method.
      mlir::IRMapping mapper;
      mapper.map(lambda->templateOp.getBody()->getArgument(0),
                 tensorFusionEnabledOn);
      OpBuilder b{computeBlock, computeBlock->begin()};

      // Copy the param region into the body.
      auto newLambda = cast<KGEN::ParamDeclareRegionOp>(
          b.clone(*lambda->canonicalLambda, mapper));

      // Rebind the parameters of the lambda from the `self` argument in the
      // method onto the specific parameters of the tensor being used at the
      // callsite.
      DenseMap<ParamDeclRefAttr, TypedAttr> paramRebinds;

      for (auto [localParamRef, methodParamDecl] :
           llvm::zip(enableFusionFunc.getCallee().getParamValues(),
                     lambda->templateOp.getInputParams())) {
        auto methodDeclRef = ParamDeclRefAttr::get(methodParamDecl.getName(),
                                                   methodParamDecl.getType());
        paramRebinds[methodDeclRef] = localParamRef;
      }

      // Update the parameter attributes.
      mlir::AttrTypeReplacer walker;
      walker.addReplacement(
          [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
            auto itr = paramRebinds.find(attr);
            if (itr != paramRebinds.end())
              return itr->second;
            return std::nullopt;
          });
      walker.recursivelyReplaceElementsIn(newLambda, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
      newLambda.setParamDeclAttr(ParamDeclAttr::get(
          newLambdaName, newLambda.getParamDecl().getType()));

      // The parameter reference of the lambda.
      auto refToLambda =
          ParamDeclRefAttr::get(newLambda.getParamDecl().getName(),
                                newLambda.getParamDecl().getType());

      // Any call to the original lambda.
      paramRebinds[ParamDeclRefAttr::get(
          lambda->canonicalLambda.getParamDecl().getName(),
          lambda->canonicalLambda.getParamDecl().getType())] = refToLambda;

      // Clone the call in the mojo template function which tells us how to
      // rebind the tensor type and remap it onto the new type.
      OwningOpRef<CallOp> newSampleCall = lambda->callUsingLambda.clone();
      walker.recursivelyReplaceElementsIn(*newSampleCall,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      size_t lambdaIndex =
          isInput ? MOGG::MOGGTensorParamAccessor::INPUT_LAMBDA_IDX
                  : MOGG::MOGGTensorParamAccessor::OUTPUT_LAMBDA_IDX;
      auto newLambdaBinding =
          newSampleCall->getCallee().getParamValues()[lambdaIndex];

      // We now have the old binding to None for the output and the new
      // binding to the the input lambda we just cloned. We now need to
      // replace all uses of the old one with the new.
      if (isInput) {
        ParamDeclRefAttr oldLambdaBinding = cast<ParamDeclRefAttr>(
            enableFusionFunc.getCallee().getParamValues()
                [MOGG::MOGGTensorParamAccessor::INPUT_LAMBDA_IDX]);

        paramRebinds[oldLambdaBinding] = newLambdaBinding;
        for (Operation &topLevelOp : gen.getOps()) {
          if (&topLevelOp != newLambda) {
            topLevelOp.walk([&](Operation *op) {
              walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                  /*replaceLocs=*/false,
                                                  /*replaceTypes=*/true);
            });
          }
        }
      } else {
        // Update all the call sites which use the output tensor to reflect
        // the new lambda.
        for (Operation *user : outputTensor.getUsers()) {
          if (auto call = dyn_cast<CallOp>(user)) {

            // Don't include any within the lambda itself.
            bool inLambda = false;
            Operation *parent = call->getParentOp();
            while (parent) {
              if (parent == newLambda) {
                inLambda = true;
                break;
              }
              parent = parent->getParentOp();
            }
            if (inLambda)
              continue;

            auto paramUpdate = [&](const MOGG::MOGGTensorParamAccessor &tensor,
                                   SmallVector<TypedAttr> &newParams) {
              if (KGEN::ParamIndexRefAttr param = tensor.outputLambdaAsRef())
                newParams[param.getIndex()] = newLambdaBinding;
            };
            rewriteCallWithNewParams(call, symTab, paramUpdate);
          }
        }
      }
    }

    // Clean up the enable fusion calls
    for (auto call : enableFusionFuncs) {
      // Theoretically the user could reference the none return...
      if (call.getResult(0).use_empty())
        call->erase();
    }
  }

  GeneratorOp sliceShapeFunction(Value shape, GeneratorOp funcToSlice) {
    Operation *op = shape.getDefiningOp();

    // Identify all ops used to create the shape.
    DenseSet<Operation *> opsToClone;
    DenseSet<Value> usedInputs;
    SmallVector<Operation *> worklist;
    worklist.push_back(op);
    while (!worklist.empty()) {
      Operation *opToProcess = worklist.back();
      worklist.pop_back();
      if (opsToClone.contains(opToProcess))
        continue;
      opsToClone.insert(opToProcess);

      for (Value operand : opToProcess->getOperands()) {
        if (operand.getDefiningOp())
          worklist.push_back(operand.getDefiningOp());
        else
          usedInputs.insert(operand);
      }
    }

    std::optional<LIT::LITSignatureType> oldLitSig = getSourceSig(funcToSlice);
    FunctionType funcTy = funcToSlice.getFunctionType();
    FunctionType litFuncType = oldLitSig->getValues();
    LIT::FnMetadataAttr oldMetadata = oldLitSig->getMetadata();

    // We need to build LIT/KGEN function types for the new signature.
    SmallVector<Type> funcInputs, litFuncInputs;
    SmallVector<StringAttr> litArgNames;

    // Identify which of the inputs we are using. This needs to be preserved
    // so MO can know which are used and in which order to map back onto the
    // original kernel.
    DenseSet<size_t> inputIndices;
    SmallVector<int32_t> usedInputsInOrder;
    for (auto [idx, operand] :
         llvm::enumerate(funcToSlice.getBody()->getArguments())) {
      if (usedInputs.contains(operand)) {
        inputIndices.insert(idx);
        funcInputs.push_back(funcTy.getInputs()[idx]);
        litFuncInputs.push_back(litFuncType.getInputs()[idx]);
        litArgNames.push_back(oldMetadata.getArgNames()[idx]);

        // Subtract one to account for the pass by ref output.
        usedInputsInOrder.push_back(static_cast<int32_t>(idx - 1));
      }
    }

    mlir::IRMapping mapper;

    GeneratorOp slicedShapeFunction = funcToSlice.cloneWithoutRegions();

    // Create the new function type.
    FunctionType newFuncType =
        FunctionType::get(ctx, funcInputs, {shape.getType()});

    SignatureType oldSig = slicedShapeFunction.getSignature();

    // Drop the input conventions.
    SignatureType newSig = SignatureType::remapToSignature(
        slicedShapeFunction.getInputParams(),
        slicedShapeFunction.getResultParams(), newFuncType,
        /*inputConventions=*/{}, oldSig.getFnEffects(),
        /*metadata*/ {});

    // Replace the parameter refs with their actual values.
    newSig = newSig.getWithValuesReplaced(newFuncType);

    // We also have to rebuild the lit metadata function so we can still infer
    // each parameter.
    FunctionType newLitFunctionType =
        FunctionType::get(ctx, litFuncInputs, {shape.getType()});

    SmallVector<KGEN::LIT::PassingKind> passingKinds{
        litFuncInputs.size(), KGEN::LIT::PassingKind::PosOnly};

    auto metadata = LIT::FnMetadataAttr::get(
        ctx, litArgNames, passingKinds, oldMetadata.getParamNames(),
        oldMetadata.getParamPassingKinds(),
        /*defaultPosArgs*/ {}, oldMetadata.getDefaultPosParams(),
        oldMetadata.getDefaultKwOnlyArgs(),
        oldMetadata.getDefaultKwOnlyParams(),
        /*numImplicitLifetimeDecls*/ 0);

    auto newLitSig = LIT::LITSignatureType::get(
        newLitFunctionType, oldLitSig->getInputParamTypes(),
        oldLitSig->getResultParamTypes(),
        /*inputConventions=*/{}, oldSig.getFnEffects(), metadata);

    slicedShapeFunction.setSignature(newSig);
    slicedShapeFunction.setFunctionType(newSig.getValues());

    slicedShapeFunction.setSourceSignatureAttr(
        PreservedAttr::get(ctx, TypeAttr::get(newLitSig)));

    Block &shapeFuncBlock =
        slicedShapeFunction.getCallableRegion()->emplaceBlock();

    // Add the operands onto the sliced shape func.
    for (auto [idx, operand] :
         llvm::enumerate(funcToSlice.getBody()->getArguments())) {
      if (inputIndices.contains(idx)) {
        Value newV =
            shapeFuncBlock.addArgument(operand.getType(), operand.getLoc());
        mapper.map(operand, newV);
      }
    }

    OpBuilder b{&shapeFuncBlock, shapeFuncBlock.begin()};

    // Clone all the ops used in the construction of the slice.
    for (Operation *op : opsToClone)
      b.clone(*op, mapper);
    b.create<KGEN::ReturnOp>(shape.getLoc(), mapper.lookup(shape));

    // Attached the used inputs as attribute. This allows the graph compiler to
    // discard unused inputs on the op.
    SmallVector<NamedAttribute> newAttrs;
    for (NamedAttribute attr : slicedShapeFunction->getAttrs())
      newAttrs.push_back(attr);
    newAttrs.push_back(NamedAttribute{b.getStringAttr("_used_inputs"),
                                      b.getI32ArrayAttr(usedInputsInOrder)});
    slicedShapeFunction->setAttrs(newAttrs);

    // Even though fusion is not presently supported on shape functions we
    // still need to add the empty annotations for consistency with the
    // compute functions.
    SmallVector<std::string> emptyInputLambdas;
    emptyInputLambdas.resize(usedInputsInOrder.size(), "");

    // Shape function still needs some metadata to inform the graph compiler
    // about its properties.
    attachMetadataToGenerator(slicedShapeFunction, emptyInputLambdas, {},
                              nullptr, nullptr, false);

    return slicedShapeFunction;
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ctx = mod.getContext();
    SymbolTable symTab{mod};
    // Scan the generators to find the global helper functions we will need to
    // call or inspect.
    for (GeneratorOp func : mod.getOps<GeneratorOp>()) {
      auto lambda = [&](TypedAttr decorator, StringRef decoratorName,
                        SmallVector<TypedAttr> &attrsToCopy) {
        if (decoratorName.starts_with(tensorInputFusionHook))
          inLambdaTemplate = LambdaTemplate{func};
        else if (decoratorName.starts_with(tensorOutputFusionHook))
          outLambdaTemplate = LambdaTemplate{func};
      };
      forEachDecorator(func, lambda);
    }

    DenseSet<GeneratorOp> seenFuncs;

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {

      if (seenFuncs.contains(userKernel))
        continue;

      std::optional<AnnotatedKernel> kernelMetadata =
          checkForMOGGAttrs(userKernel);
      if (!kernelMetadata.has_value())
        continue;

      // If there are no tensors detected on the API then it's not a new API
      // kernel.
      // TODO: This should be removed when there's only one API.
      if (!hasAtLeastOneTensor(userKernel))
        continue;

      // Currently we only support kernels which return something. We will later
      // enforce that this is a tensor.
      if (!userKernel.getSignature().hasMemoryOnlyResult())
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      KGEN::GeneratorOp slicedComputeFunction = userKernel.clone();

      // Search for any function which allocates a new tensor and a move from
      // that into one of the input operands (meaning it is actually an output).
      KGEN::CallOp allocationFunc, constructor;

      // If this is an elementwise kernel we are expecting to see a call to the
      // elementwise generator.
      KGEN::CallOp elementwiseOp;

      // If the user has any call to enable fusion then we turn on fusion for
      // that tensor.
      SmallVector<KGEN::CallOp> enableFusionFuncs;
      SmallVector<KGEN::CallOp> deconstructors;

      std::optional<MOGG::MOGGTensorParamAccessor> outTensorParameters =
          getTensorRepFromFunctionInput(slicedComputeFunction, 0);
      if (!outTensorParameters.has_value())
        continue;

      // Scan the kernel and identify the callsites of annotated functions that
      // we can understand.
      for (KGEN::CallOp call : slicedComputeFunction.getOps<KGEN::CallOp>()) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          continue;

        auto identifyCalls = [&](TypedAttr decorator, StringRef decoratorName,
                                 SmallVector<TypedAttr> &attrsToCopy) {
          if (decoratorName.starts_with(tensorAllocDecorator)) {
            allocationFunc = call;
          } else if (decoratorName.starts_with(tensorCopyConstructDecorator)) {
            constructor = call;
          } else if (decoratorName.starts_with(tensorEnableFusion)) {
            enableFusionFuncs.push_back(call);
          } else if (decoratorName.starts_with(elementwiseHook)) {
            elementwiseOp = call;
          } else if (decoratorName.starts_with(tensorDeconstructDecorator)) {
            deconstructors.push_back(call);
          }
        };
        forEachDecorator(func, identifyCalls);
      }

      // Strip all debug info. Its too annoying to maintain and there is no way
      // to actually debug the sliced kernel directly. Users would debug the
      // base kernel.
      slicedComputeFunction.walk(
          [](DebugInfo::ValueOp debug) { debug.erase(); });

      // Clean up all the deconstructors. Not strictly needed as they will be
      // elaborated with the ref counting / allocation off.
      for (KGEN::CallOp deconstruct : deconstructors)
        deconstruct.erase();

      // Resize the input and output lambda metadata to encapsulate all inputs
      // and outputs. Each one is marked off. As we detect in / out lambdas we
      // will populate these.
      SmallVector<std::string> inputLambdaNames, outputLambdaNames;
      inputLambdaNames.resize(userKernel.getBody()->getArguments().size() - 1,
                              "");
      outputLambdaNames.resize(1, "");

      // Output tensor is the first argument.
      Value outputTensor = slicedComputeFunction.getBody()->getArgument(0);
      Value shape;

      bool isView = false;

      // Any MOGG annotated kernel which has no allocation should be treated as
      // a view.
      if (!allocationFunc) {
        isView = true;

        // Even if a view op calls this it should not appear as an elementwise
        // kernel.
        elementwiseOp = nullptr;
      } else if (!constructor) {
        // Exit and clean up if the kernel is not what we expect. Allocators and
        // constructors are expected to appear as a pair.
        slicedComputeFunction.erase();
        continue;
      } else {
        // Otherwise we are dealing with a normal allocating op.
        shape = allocationFunc.getOperand(1);

        // clang-format-off
        // Functions with tensor allocation will follow the rough pattern.
        // fn (*tensor):
        //   tmp = allocate(...)
        //   ...
        //   copy_construct(tensor, tmp)
        // clang-format-on
        // We can use this to identify the output tensor and the the tensor
        // which has been allocated. To us they are an alias.
        Value tmpTensor = constructor.getOperand(1);
        tmpTensor.replaceAllUsesWith(outputTensor);

        // Remove the allocation and assignment from the sliced compute
        // function.
        constructor.erase();
        allocationFunc.erase();

        // Turn on fusion for any tensors which have set fusion.
        enableFusion(slicedComputeFunction, outputTensor, inputLambdaNames,
                     outputLambdaNames, enableFusionFuncs, symTab);
      }

      // Any `none` parameters need to be instantiated since MOGG can't provide
      // them. These are the lambdas which have now been turned off.
      instantiateNoneParamLambdas(slicedComputeFunction);

      // Override the ref counting so all tensors are owned by the graph
      // compiler.
      markAllTensorsAsOwned(slicedComputeFunction, symTab);

      // Add compute function part to the module, i.e the kernel sans
      // allocation.
      symTab.insert(slicedComputeFunction);

      GeneratorOp slicedShapeFunction;

      if (shape) {
        slicedShapeFunction = sliceShapeFunction(shape, slicedComputeFunction);
      } else {
        // TODO: Shape functions for views.
      }

      if (slicedShapeFunction) {
        // Has to be added to the module symbol table otherwise the function
        // name will not match in the metadata.
        symTab.insert(slicedShapeFunction);

        // Drop the decorators on the shape function.
        slicedShapeFunction.setDecorators({});
        seenFuncs.insert(slicedShapeFunction);
      }

      // Add info for mogg to read off the kernel.
      attachMetadataToGenerator(slicedComputeFunction, inputLambdaNames,
                                outputLambdaNames, elementwiseOp,
                                slicedShapeFunction, isView);

      // Remove the attributes from the user kernel as that should remain
      // untouched for the user to use directly in their code.
      userKernel.setDecorators(
          KGEN::DecoratorsAttr::get(ctx, kernelMetadata->nonMOGGDecorators));

      // Don't process the function we just added if we see it again.
      seenFuncs.insert(slicedComputeFunction);
    }

    // mod.walk([&](GeneratorOp gen) { gen.removeSourceSignatureAttr(); });
  }
};

} // namespace
