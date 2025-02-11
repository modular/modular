//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
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

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEMOGGFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {
class SliceMOGGFuncsPass
    : public MOGGPreElab::impl::SliceMOGGFuncsBase<SliceMOGGFuncsPass> {
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
    }
    /// The op we are pulling this info from.
    GeneratorOp templateOp;

    /// This the the template lambda we will clone as the input or output
    /// lambda.
    ParamDeclareRegionOp canonicalLambda;
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
    llvm::append_range(newAttrs, gen->getAttrs());

    Builder b{ctx};

    // Mark this as a sliced function so MOGG lowering can identify it.
    newAttrs.push_back(
        NamedAttribute{b.getStringAttr(SLICED_ATTR), b.getUnitAttr()});

    // Attach the lambda metadata.
    SmallVector<StringRef> inNames =
        llvm::to_vector_of<StringRef>(inputLambdaNames);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGInputLambdas),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames =
        llvm::to_vector_of<StringRef>(outputLambdaNames);
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
                    ArrayRef<unsigned> fusedOperands) {
    Block *computeBlock = gen.getBody();

    ArrayAttr argumentTypeNames =
        cast<ArrayAttr>(gen->getAttr(MOGG_ARG_TYPE_NAMES));

    ArrayAttr argsParams =
        dyn_cast_or_null<ArrayAttr>(gen->getAttr(MOGG_TENSOR_ARG_PARAMS));
    if (!argsParams)
      return;

    for (unsigned idx : fusedOperands) {
      Value tensorFusionEnabledOn = gen.getBody()->getArgument(idx);

      bool isInput = idx > 0;
      auto typeNameAttr =
          dyn_cast<StringAttr>(argumentTypeNames.getValue()[idx]);
      bool isVariadic =
          typeNameAttr && typeNameAttr.getValue() == MOJO_STATIC_INT_TUPLE_NAME;
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

      OpBuilder builder{computeBlock, computeBlock->begin()};

      if (isVariadic) {
        // The fused argument is a StaticTuple (array) of tensors.
        // We load the tensor at index 0 here (In MoToMOGG 0 will be replaced
        // with the right index for each input lambda). We insert the load
        // outside of the lambda first, then move it inside later (to ensure
        // mapper works correctly).
        tensorFusionEnabledOn = builder.create<KGEN::POP::ArrayGetOp>(
            tensorFusionEnabledOn.getLoc(), tensorFusionEnabledOn, 0);
      }

      mapper.map(lambda->templateOp.getBody()->getArgument(0),
                 tensorFusionEnabledOn);

      // Copy the param region into the body.
      auto newLambda = cast<ParamDeclareRegionOp>(
          builder.clone(*lambda->canonicalLambda, mapper));

      if (isVariadic) {
        // Move the ArrayGetOp at the beginning of the block.
        Operation *getTensorOp = tensorFusionEnabledOn.getDefiningOp();
        getTensorOp->moveBefore(&newLambda.getBodyRegion().front(),
                                newLambda.getBodyRegion().front().begin());
      }

      SmallVector<TypedAttr> remappedLambdaParams;
      auto argsParamsDict = cast<DictionaryAttr>(argsParams[idx]);

      // Get all input parameters from the template
      auto inputParams = lambda->templateOp.getInputParams();

      // For each input parameter, find its matching value in the dictionary
      for (auto paramDecl : inputParams) {
        auto paramName = paramDecl.getName();
        ASSERT_STREAM(paramName, "Parameter must have a name");

        StringRef demangledName =
            LIT::demangleParameterName(paramName.getValue());

        auto paramValue = argsParamsDict.get(demangledName);
        ASSERT_STREAM(paramValue, "Missing parameter '"
                                      << demangledName
                                      << "' in arguments dictionary");

        auto typedValue = dyn_cast<TypedAttr>(paramValue);
        ASSERT_STREAM(typedValue, "Parameter value must be a TypedAttr");

        remappedLambdaParams.push_back(typedValue);
      }

      // Rebind the parameters of the lambda from the `self` argument in the
      // method onto the specific parameters of the tensor being used at the
      // callsite.
      ParameterEvaluator evaluator;

      for (auto [localParamRef, methodParamDecl] :
           llvm::zip(remappedLambdaParams, inputParams)) {
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

    DenseSet<GeneratorOp> seenFuncs;
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
      /// TODO: GEX-1046: We should have markers in Mojo for what is an input
      /// and what is an output (ex: mo.top_k).
      unsigned kernelOutputsCount = 1;
      for (size_t i = 1, e = argumentTypeNames.getValue().size(); i < e; i++) {
        auto nameAttr = dyn_cast<StringAttr>(argumentTypeNames.getValue()[i]);
        if (nameAttr &&
            (nameAttr.getValue() == MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME ||
             nameAttr.getValue() == MOJO_STATIC_INT_TUPLE_NAME)) {
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
                   fusedOperands);

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
