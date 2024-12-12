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
#include "UserLibraryChecker.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEMOGGFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {
class SliceMOGGFuncsPass
    : public M::KGEN::MOGGPreElab::impl::SliceMOGGFuncsBase<
          SliceMOGGFuncsPass> {
private:
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
                                 bool isView = false) {
    // The new attributes on the generator.
    SmallVector<NamedAttribute> newAttrs;

    // Add all the old attributes.
    for (NamedAttribute attr : gen->getAttrs())
      newAttrs.push_back(attr);

    OpBuilder b{ctx};

    // Mark this as a sliced function so MOGG lowering can identify it.
    newAttrs.push_back(
        NamedAttribute{b.getStringAttr(SLICED_ATTR), b.getUnitAttr()});

    if (isView) {
      newAttrs.push_back(
          NamedAttribute{b.getStringAttr("_view_op"), b.getUnitAttr()});
    }

    // Attach the lambda metadata.
    SmallVector<StringRef> inNames;
    for (const std::string &name : inputLambdaNames)
      inNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGInputLambdas),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames;
    for (const std::string &name : outputLambdaNames)
      outNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGOutputLambdas),
                                      b.getStrArrayAttr(outNames)});
    gen->setAttrs(newAttrs);
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
      if (func->hasAttr(Decorators::INPUT_FUSION.attr))
        inLambdaTemplate = LambdaTemplate{func};
      else if (func->hasAttr(Decorators::OUTPUT_FUSION.attr))
        outLambdaTemplate = LambdaTemplate{func};
    }

    DenseSet<GeneratorOp> seenFuncs;

    auto checker = UserLibraryChecker(mod, symTab);
    if (failed(checker.run())) {
      signalPassFailure();
      return;
    }

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {

      if (seenFuncs.contains(userKernel))
        continue;

      // Skip non kernels.
      if (!isKernel(userKernel))
        continue;

      // Currently we only support kernels which return something. We will
      // later enforce that this is a tensor.
      if (!userKernel.getSignatureGenerator().getBody().hasMemoryOnlyResult())
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      KGEN::GeneratorOp slicedComputeFunction = userKernel.clone();
      std::string name =
          (Twine(userKernel.getSymName()) + Twine("_COMPUTE")).str();
      slicedComputeFunction.setSymName(name);

      // Search for any function which allocates a new tensor and a move from
      // that into one of the input operands (meaning it is actually an
      // output).
      KGEN::CallOp allocationFunc, constructor;

      // If the user has any call to enable fusion then we turn on fusion for
      // that tensor.
      SmallVector<KGEN::CallOp> enableFusionFuncs;
      SmallVector<KGEN::CallOp> deconstructors;

      // Scan the kernel and identify the callsites of annotated functions
      // that we can understand.
      for (KGEN::CallOp call : slicedComputeFunction.getOps<KGEN::CallOp>()) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          continue;

        if (func->hasAttr(Decorators::ENABLE_FUSION.attr))
          enableFusionFuncs.push_back(call);
      }

      // Strip all debug info. Its too annoying to maintain and there is no
      // way to actually debug the sliced kernel directly. Users would debug
      // the base kernel.
      DebugInfo::stripDebugInfo(slicedComputeFunction,
                                /*preserveLineTables=*/true);

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

      // Output tensor is the last argument.
      assert(slicedComputeFunction.getSignatureGenerator()
                 .getBody()
                 .hasMemoryOnlyResult());
      Value outputTensor =
          slicedComputeFunction.getBody()->getArguments().back();

      bool isView = false;
      // Any MOGG annotated kernel which has no allocation should be treated
      // as a view.
      if (!allocationFunc) {
        isView = true;
      } else if (!constructor) {
        // Exit and clean up if the kernel is not what we expect. Allocators
        // and constructors are expected to appear as a pair.
        slicedComputeFunction.erase();
        continue;
      } else {
        // Erase any lifetime markers on the temporary if it has them.
        Value tmpTensor = constructor.getOperand(1);
        auto alloc = tmpTensor.getDefiningOp<POP::StackAllocationOp>();
        if (alloc && alloc.getMarkedLifetimes()) {
          for (Operation *user :
               llvm::make_early_inc_range(tmpTensor.getUsers())) {
            if (isa<POP::StackAllocLifetimeStartOp,
                    POP::StackAllocLifetimeEndOp>(user))
              user->erase();
          }
        }

        // Functions with tensor allocation will follow the rough pattern.
        //
        // ```
        // fn (*tensor):
        //   tmp = allocate(...)
        //   ...
        //   copy_construct(tensor, tmp)
        // ```
        //
        // We can use this to identify the output tensor and the the tensor
        // which has been allocated. To us they are an alias.
        tmpTensor.replaceAllUsesWith(outputTensor);

        // Remove the allocation and assignment from the sliced compute
        // function.
        constructor.erase();
        allocationFunc.erase();
      }

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
                                outputLambdaNames, isView);

      // Remove the kernel attr from the user kernel.
      userKernel->removeAttr(kernelRegistrationAttr);

      //  Don't process the function we just added if we see it again.
      seenFuncs.insert(slicedComputeFunction);
    }
  }
};
} // namespace
