//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/Transforms/StripDebugInfo.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "Helpers.h"
#include "Support/AssertStream.h"
#include "UserLibraryChecker.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEMOGGDPSFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {
class SliceMOGGDPSFuncsPass
    : public MOGGPreElab::impl::SliceMOGGDPSFuncsBase<SliceMOGGDPSFuncsPass> {
private:
  /// We have a special mojo hook which show us what the canonical lambda
  /// looks like and a call which tells us the resulting type with the lambda
  /// applied.
  struct LambdaTemplate {
    LambdaTemplate() = default;

    /// Scan the hook for the properties that we know exist.
    LambdaTemplate(GeneratorOp hook) : templateOp(hook) {
      for (auto region : hook.getOps<ParamDeclareRegionOp>()) {
        ASSERT_STREAM(
            canonicalLambda == nullptr,
            "there must be only one region in the I/O lambda intrinsic");
        canonicalLambda = region;
      }
      ASSERT_STREAM(canonicalLambda != nullptr,
                    "missing region in the I/O lambda intrinsic");

      for (auto call : hook.getOps<CallOp>()) {
        ASSERT_STREAM(
            callUsingLambda == nullptr,
            "there must be only one CallOp in the I/O lambda intrinsic");
        callUsingLambda = call;
      }
      ASSERT_STREAM(callUsingLambda != nullptr,
                    "missing CallOp in the I/O lambda intrinsic");
    }
    /// The op we are pulling this info from.
    GeneratorOp templateOp;

    /// This the the template lambda we will clone as the input or output
    /// lambda.
    ParamDeclareRegionOp canonicalLambda;

    /// This call shows us how the lambda needs to be bound.
    CallOp callUsingLambda;
  };

  MLIRContext *ctx;

  /// The reference input and output lambdas we should use for materializing the
  /// input/output fusion.
  LambdaTemplate inLambdaTemplate, outLambdaTemplate;

  void attachMetadataToGenerator(GeneratorOp gen,
                                 ArrayRef<std::string> inputLambdaNames,
                                 ArrayRef<std::string> outputLambdaNames) {
    // The new attributes on the generator.
    SmallVector<NamedAttribute> newAttrs;

    // Add all the old attributes.
    for (NamedAttribute attr : gen->getAttrs())
      newAttrs.push_back(attr);

    OpBuilder b{ctx};

    // Mark this as a sliced function so MOGG lowering can identify it.
    newAttrs.push_back(
        NamedAttribute{b.getStringAttr(SLICED_ATTR), b.getUnitAttr()});

    // Attach the lambda metadata.
    SmallVector<StringRef> inNames;
    for (StringRef name : inputLambdaNames)
      inNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGInputLambdas),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames;
    for (StringRef name : outputLambdaNames)
      outNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGOutputLambdas),
                                      b.getStrArrayAttr(outNames)});
    gen->setAttrs(newAttrs);
  }

  /// We enable fusion for I/Os tensors that have opted in.
  /// Enabling fusion involves materializing a call to the input/output lambda
  /// within the body of the function and building a new tensor spec parameter
  /// with the lambda replaced.
  void enableFusion(GeneratorOp gen,
                    MutableArrayRef<std::string> inputLambdaNames,
                    MutableArrayRef<std::string> outputLambdaNames,
                    SymbolTable &symTab, ArrayRef<unsigned> fusedOperands) {
    Block *computeBlock = gen.getBody();

    ArrayAttr tensorSpecParams = dyn_cast_or_null<ArrayAttr>(
        gen->getAttr(kKernelTensorSpecParameterAttrName));
    if (!tensorSpecParams)
      return;

    ArrayAttr argsParams =
        dyn_cast_or_null<ArrayAttr>(gen->getAttr(MOGG_ARG_PARAMS));
    if (!argsParams)
      return;

    for (unsigned idx : fusedOperands) {
      Value tensorFusionEnabledOn = gen.getBody()->getArgument(idx);

      if (idx >= tensorSpecParams.size())
        continue;
      auto argTensorSpecParam =
          dyn_cast<KGEN::ParamDeclAttr>(tensorSpecParams[idx]);
      if (!argTensorSpecParam)
        continue;

      ParamDeclRefAttr tensorSpecRef =
          ParamDeclRefAttr::get(argTensorSpecParam);

      bool isInput = idx > 0;
      ASSERT_STREAM(inLambdaTemplate.templateOp && outLambdaTemplate.templateOp,
                    "intrinsic I/O fusion hooks not found");
      LambdaTemplate *lambda = isInput ? &inLambdaTemplate : &outLambdaTemplate;
      std::string newLambdaName =
          isInput ? "input_" + std::to_string(idx) + "_fn" : "output_0_fn";
      if (isInput)
        inputLambdaNames[idx - 1] = newLambdaName;
      else
        outputLambdaNames[0] = newLambdaName;

      // Instead of referring to the `self` argument of the wrapper function
      // which contains the canonical lambda we remap onto the argument of
      // this function which invoked the enable fusion method.
      IRMapping mapper;
      ASSERT_STREAM(lambda->templateOp != nullptr, "missing lambda");

      mapper.map(lambda->templateOp.getBody()->getArgument(0),
                 tensorFusionEnabledOn);
      OpBuilder b{computeBlock, computeBlock->begin()};

      // Copy the param region into the body.
      auto newLambda =
          cast<ParamDeclareRegionOp>(b.clone(*lambda->canonicalLambda, mapper));

      SmallVector<TypedAttr> remappedLambdaParams;
      for (auto attr : cast<ArrayAttr>(argsParams[idx]).getValue())
        remappedLambdaParams.push_back(cast<TypedAttr>(attr));
      // The method has an extra param_spec argument.
      remappedLambdaParams.push_back(tensorSpecRef);
      ASSERT_STREAM(remappedLambdaParams.size() ==
                        lambda->templateOp.getInputParams().size(),
                    << "parameters count mismatch");

      // Rebind the parameters of the lambda from the `self` argument in the
      // method onto the specific parameters of the tensor being used at the
      // callsite.
      ParameterEvaluator evaluator;

      for (auto [localParamRef, methodParamDecl] : llvm::zip(
               remappedLambdaParams, lambda->templateOp.getInputParams())) {
        auto methodDeclRef = ParamDeclRefAttr::get(methodParamDecl.getName(),
                                                   methodParamDecl.getType());
        evaluator.setParameterValue(methodDeclRef.getName(), localParamRef);
      }

      // Update the parameter attributes.
      mlir::AttrTypeReplacer walker;
      walker.addReplacement([&](ParamDeclRefAttr attr) {
        return evaluator.getReboundAttribute(attr);
      });
      walker.recursivelyReplaceElementsIn(newLambda, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
      newLambda.setParamDeclAttr(ParamDeclAttr::get(
          newLambdaName, newLambda.getParamDecl().getType()));

      // Now that we have the lambda parameter,
      // We need to rebuild a new StaticTensorSpec.

      // The parameter reference of the lambda.
      auto refToLambda =
          ParamDeclRefAttr::get(newLambda.getParamDecl().getName(),
                                newLambda.getParamDecl().getType());

      // Any call to the original lambda.
      evaluator.setParameterValue(
          lambda->canonicalLambda.getParamDecl().getName(), refToLambda);

      // Clone the call in the mojo template function which tells us how to
      // rebuild the tensor spec
      OwningOpRef<CallOp> newSampleCall = lambda->callUsingLambda.clone();
      walker.recursivelyReplaceElementsIn(*newSampleCall,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

      auto newTensorSpecBinding =
          newSampleCall->getCallee().getParamValues().back();
      ASSERT_STREAM(tensorSpecRef.getType() == newTensorSpecBinding.getType(),
                    << "invalid type");

      // Now replace all uses of the tensor spec in the IR.
      ParameterEvaluator fixIREvaluator;
      fixIREvaluator.setParameterValue(tensorSpecRef.getName(),
                                       newTensorSpecBinding);
      mlir::AttrTypeReplacer fixIRWalker;
      fixIRWalker.addReplacement([&](ParamDeclRefAttr attr) {
        return fixIREvaluator.getReboundAttribute(attr);
      });

      for (Operation &op : gen.getBody()->getOperations()) {
        if (&op != newLambda) {
          fixIRWalker.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/true,
                                                   /*replaceLocs=*/false,
                                                   /*replaceTypes=*/true);
        }
      };
    }
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ctx = mod.getContext();
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symTab = analysis.getTopLevelSymbolTable();

    // Scan the generators to find the global helper functions we will need to
    // call or inspect.
    for (GeneratorOp func : mod.getOps<GeneratorOp>()) {
      if (func->hasAttr(MOGG_INTRINSIC_INPUT_FUSION_HOOK))
        inLambdaTemplate = LambdaTemplate{func};
      else if (func->hasAttr(MOGG_INTRINSIC_OUTPUT_FUSION_HOOK))
        outLambdaTemplate = LambdaTemplate{func};
    }

    auto checker = UserLibraryChecker(mod, symTab);
    if (failed(checker.run())) {
      signalPassFailure();
      return;
    }

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {
      // Only look at the DPS execute functions.
      if (!userKernel->hasAttr(kMOGGExecuteFunctionLabel))
        continue;

      // Don't process again functions that we already sliced.
      if (userKernel->hasAttr(SLICED_ATTR))
        continue;

      // Dummy execute function without operands (mo.reshape).
      /// TODO: this should not be allowed for users.
      ArrayAttr argumentTypeNames =
          dyn_cast_or_null<ArrayAttr>(userKernel->getAttr(MOGG_ARG_TYPE_NAMES));
      if (!argumentTypeNames)
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      GeneratorOp slicedComputeFunction = userKernel.clone();
      std::string name = (Twine(userKernel.getSymName()) + "_COMPUTE").str();
      slicedComputeFunction.setSymName(name);

      // Figure out the number of tensor inputs for the kernel.
      // The first operand is always the output.
      // And there might be non-tensor arguments (eg MojoCallContext).
      unsigned kernelInputsCount = 0;
      /// TODO: GRA-1046: We should have markers in Mojo for what is an input
      /// and what is an output (ex: mo.top_k).
      unsigned kernelOutputsCount = 1;
      for (size_t i = 1, e = argumentTypeNames.getValue().size(); i < e; i++) {
        auto nameAttr = dyn_cast<StringAttr>(argumentTypeNames.getValue()[i]);
        if (nameAttr &&
            nameAttr.getValue() == MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME) {
          ++kernelInputsCount;
        }
      }

      // Figure out which kernel input / outputs are fused.
      SmallVector<unsigned> fusedOperands;
      if (userKernel->hasAttr(kMOGGElementFunction)) {
        // For elementwise kernel, fuse everything.
        for (unsigned i = 0, e = kernelInputsCount + kernelOutputsCount; i < e;
             ++i)
          fusedOperands.push_back(i);
      } else if (auto fusedOperandsAttr = dyn_cast_or_null<ArrayAttr>(
                     userKernel->getAttr(kMOGGFusableArgs))) {
        for (Attribute attr : fusedOperandsAttr.getValue())
          fusedOperands.push_back(
              cast<IntegerAttr>(attr).getValue().getZExtValue());
      }

      // Resize the input and output lambda metadata to encapsulate all inputs
      // and outputs. Each one is marked off. As we detect in / out lambdas we
      // will populate these.
      SmallVector<std::string> inputLambdaNames, outputLambdaNames;
      inputLambdaNames.resize(kernelInputsCount, "");
      outputLambdaNames.resize(kernelOutputsCount, "");

      enableFusion(slicedComputeFunction, inputLambdaNames, outputLambdaNames,
                   symTab, fusedOperands);

      // Strip all debug info. Its too annoying to maintain and there is no
      // way to actually debug the sliced kernel directly. Users would debug
      // the base kernel.
      DebugInfo::stripDebugInfo(slicedComputeFunction,
                                /*preserveLineTables=*/true);

      // Add compute function part to the module, i.e the kernel sans
      // allocation.
      symTab.insert(slicedComputeFunction);

      // Add info for mogg to read off the kernel.
      attachMetadataToGenerator(slicedComputeFunction, inputLambdaNames,
                                outputLambdaNames);

      // Remove attributes that would cause MojoLibraryAnalysis to treat the
      // sliced function as a registered kernel.
      userKernel->removeAttr(kernelRegistrationAttr);
      userKernel->removeAttr(kMOGGExecuteFunctionLabel);
      userKernel->removeAttr(shapeFuncRegistrationAttr);
      userKernel->removeAttr(kMOGGShapeFunctionLabel);
      userKernel->removeAttr(kMOGGPyTorchFallbackFunctionLabel);
    }
  }
};
} // namespace
