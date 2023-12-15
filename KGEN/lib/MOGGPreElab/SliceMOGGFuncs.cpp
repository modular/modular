//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
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
    "$utils::$_annotations::mogg_register";
// TODO(#27757): Temporary as transition to Mojo async/await.
constexpr StringLiteral willBecomeAsyncDecorator =
    "$utils::$_annotations::mogg_will_become_async";
constexpr StringLiteral registerOverrideDecorator =
    "$utils::$_annotations::mogg_register_override";
constexpr StringLiteral experimentalDecorator =
    "$utils::$_annotations::mogg_kgen_experiment_kernel";

constexpr StringLiteral tensorAllocDecorator =
    "$utils::$_annotations::mogg_tensor_allocator";
constexpr StringLiteral tensorMoveDecorator =
    "$utils::$_annotations::mogg_tensor_move_constructor";
constexpr StringLiteral tensorSimdStoreDecorator =
    "$utils::$_annotations::mogg_tensor_simd_store";

constexpr StringLiteral elementwiseHook =
    "$utils::$_annotations::mogg_elementwise_hook";
constexpr StringLiteral tensorEnableFusion =
    "$utils::$_annotations::mogg_enable_fusion";
constexpr StringLiteral tensorInputFusionHook =
    "$utils::$_annotations::mogg_input_fusion_hook";
constexpr StringLiteral tensorOutputFusionHook =
    "$utils::$_annotations::mogg_output_fusion_hook";

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

class SliceMOGGFuncsPass
    : public M::KGEN::MOGGPreElab::impl::SliceMOGGFuncsBase<
          SliceMOGGFuncsPass> {
private:
  struct AnnotatedKernel {
    /// Every mogg kernel should have a registration hook mapping it onto an op.
    TypedAttr moggRegister;

    /// Currently we only run on MOGG kernels with the experimental attribute
    /// attached.
    TypedAttr experimentalAttr;

    /// If true, indicates the kernel will be implemented by an 'async' Mojo
    /// function.
    bool isAsync = false;

    /// When cloning the kernel we want to preserve the decorators unrelated to
    /// mogg.
    SmallVector<TypedAttr> nonMOGGDecorators;
  };

  std::optional<AnnotatedKernel> checkForMOGGAttrs(GeneratorOp userFunc) {
    AnnotatedKernel metadata;

    // Look for the mogg attributes on the kernels.
    auto lambda = [&](TypedAttr decorator, StringRef decoratorName,
                      SmallVector<TypedAttr> &attrsToCopy) {
      if (decoratorName.startswith(experimentalDecorator)) {
        metadata.experimentalAttr = decorator;
        // Drop the mogg decorator.
        attrsToCopy.pop_back();
      } else if (decoratorName.startswith(registerDecorator) ||
                 decoratorName.startswith(registerOverrideDecorator)) {
        metadata.moggRegister = decorator;
        // Drop the mogg decorator
        attrsToCopy.pop_back();
      } else if (decoratorName.startswith(willBecomeAsyncDecorator)) {
        // TODO(#27757): Temporary while transition to Mojo async/await.
        // Eventually this will be implied by the generator op's signature.
        metadata.isAsync = true;
        // Drop the mogg decorator
        attrsToCopy.pop_back();
      }
    };

    // Capture the decorators unrelated to mogg so they can be preserved.
    metadata.nonMOGGDecorators = forEachDecorator(userFunc, lambda);

    // This is not a mogg kernel if it doesn't have a register and (for now) the
    // experimental attribute.
    if (!metadata.experimentalAttr || !metadata.moggRegister)
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

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    SymbolTable symTab{mod};

    LambdaTemplate inLambdaTemplate, outLambdaTemplate;

    // Scan the generators to find the global helper functions we will need to
    // call or inspect.
    for (GeneratorOp func : mod.getOps<GeneratorOp>()) {
      auto lambda = [&](TypedAttr decorator, StringRef decoratorName,
                        SmallVector<TypedAttr> &attrsToCopy) {
        if (decoratorName.startswith(tensorInputFusionHook))
          inLambdaTemplate = LambdaTemplate{func};
        else if (decoratorName.startswith(tensorOutputFusionHook))
          outLambdaTemplate = LambdaTemplate{func};
      };
      forEachDecorator(func, lambda);
    }

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {

      std::optional<AnnotatedKernel> kernelMetadata =
          checkForMOGGAttrs(userKernel);
      if (!kernelMetadata.has_value())
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      KGEN::GeneratorOp slicedComputeFunction = userKernel.clone();
      Block *computeBlock = slicedComputeFunction.getBody();

      // Search for any function which allocates a new tensor and a move from
      // that into one of the input operands (meaning it is actually an output).
      KGEN::CallOp allocationFunc, moveConstructor;

      // If this is an elementwise kernel we are expecting to see a call to the
      // elementwise generator.
      KGEN::CallOp elementwiseOp;

      // If the user has any call to enable fusion then we turn on fusion for
      // that tensor.
      SmallVector<KGEN::CallOp> enableFusionFuncs;
      SmallVector<KGEN::CallOp> simdStores;

      // Scan the kernel and identify the callsites of annotated functions that
      // we can understand.
      for (KGEN::CallOp call : slicedComputeFunction.getOps<KGEN::CallOp>()) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          continue;

        auto identifyCalls = [&](TypedAttr decorator, StringRef decoratorName,
                                 SmallVector<TypedAttr> &attrsToCopy) {
          if (decoratorName.startswith(tensorAllocDecorator)) {
            allocationFunc = call;
          } else if (decoratorName.startswith(tensorMoveDecorator)) {
            moveConstructor = call;
          } else if (decoratorName.startswith(tensorEnableFusion)) {
            enableFusionFuncs.push_back(call);
          } else if (decoratorName.startswith(elementwiseHook)) {
            elementwiseOp = call;
          }
        };
        forEachDecorator(func, identifyCalls);
      }

      // Exit and clean up if the kernel is not what we expect.
      if (!moveConstructor || !allocationFunc) {
        slicedComputeFunction.erase();
        continue;
      }

      // Look for simd stores recursively.
      slicedComputeFunction.walk([&](KGEN::CallOp call) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          return;

        // Look for mogg attributes on any of the calls.
        auto identifyCalls = [&](TypedAttr decorator, StringRef decoratorName,
                                 SmallVector<TypedAttr> &attrsToCopy) {
          if (decoratorName.startswith(tensorSimdStoreDecorator))
            simdStores.push_back(call);
        };
        forEachDecorator(func, identifyCalls);
      });

      Value outputTensor = moveConstructor.getOperand(0);
      Value tmpTensor = moveConstructor.getOperand(1);
      tmpTensor.replaceAllUsesWith(outputTensor);

      // Remove the allocation and assignment from the sliced compute function.
      moveConstructor.erase();
      allocationFunc.erase();

      // Resize the input and output lambda metadata to encapsulate all inputs
      // and outputs. Each one is marked off. As we detect in / out lambdas we
      // will populate these.
      SmallVector<std::string> inputLambdaNames, outputLambdaNames;
      inputLambdaNames.resize(userKernel.getBody()->getArguments().size() - 1,
                              "");
      outputLambdaNames.resize(1, "");

      for (KGEN::CallOp enableFusionFunc : enableFusionFuncs) {
        std::string newLambdaName;
        Value tensorFusionEnabledOn = enableFusionFunc.getOperand(0);

        bool isInput = true;
        LambdaTemplate *lambda;

        if (tensorFusionEnabledOn ==
            slicedComputeFunction.getBody()->getArgument(0)) {
          newLambdaName = "output_0_fn";
          outputLambdaNames[0] = newLambdaName;
          lambda = &outLambdaTemplate;
          isInput = false;
        } else {
          lambda = &inLambdaTemplate;

          // We are dealing with an input.
          for (auto [index, value] : llvm::enumerate(
                   slicedComputeFunction.getBody()->getArguments())) {
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
        walker.recursivelyReplaceElementsIn(
            *newSampleCall, /*replaceAttrs=*/true, /*replaceLocs=*/false,
            /*replaceTypes=*/true);

        // We now have the old binding to None for the output and the new
        // binding to the the input lambda we just cloned. We now need to
        // replace all uses of the old one with the new.
        if (isInput) {
          ParamDeclRefAttr oldLambdaBinding = cast<ParamDeclRefAttr>(
              enableFusionFunc.getCallee().getParamValues()[3]);
          auto newLambdaBinding =
              newSampleCall->getCallee().getParamValues()[3];

          paramRebinds[oldLambdaBinding] = newLambdaBinding;
          for (Operation &topLevelOp : slicedComputeFunction.getOps()) {
            if (&topLevelOp != newLambda) {
              topLevelOp.walk([&](Operation *op) {
                walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                    /*replaceLocs=*/false,
                                                    /*replaceTypes=*/true);
              });
            }
          }
        } else {
          TypedAttr oldLambdaBinding =
              enableFusionFunc.getCallee().getParamValues()[4];
          auto newLambdaBinding =
              newSampleCall->getCallee().getParamValues()[4];

          mlir::AttrTypeReplacer walker2;
          walker2.addReplacement(
              [&](TypedAttr attr) -> std::optional<TypedAttr> {
                if (attr == oldLambdaBinding)
                  return newLambdaBinding;
                return std::nullopt;
              });

          for (Operation &topLevelOp : slicedComputeFunction.getOps()) {
            if (&topLevelOp != newLambda) {
              topLevelOp.walk([&](Operation *op) {
                walker2.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                     /*replaceLocs=*/false,
                                                     /*replaceTypes=*/true);
              });
            }
          }
        }
      }

      int num_params = 0;
      // For now assume all variant parameters are referring to a lambda. Each
      // variant parameter will be initalized as a None type to represent no
      // fusion.
      for (auto param : slicedComputeFunction.getInputParams()) {
        if (auto asVariant = dyn_cast<KGEN::VariantType>(param.getType())) {
          OpBuilder b{computeBlock, computeBlock->begin()};

          auto lambdaNoneTy = KGEN::VariantAttr::get(
              b.getIntegerAttr(b.getI1Type(), 0), 1, asVariant);

          std::string newalias = "_none_lambda_" + std::to_string(num_params++);
          b.create<ParamDeclareOp>(
              slicedComputeFunction.getLoc(),
              ParamDeclAttr::get(newalias, param.getType()), lambdaNoneTy);

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

          slicedComputeFunction.walk([&](Operation *op) {
            if (op == slicedComputeFunction) {
              walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                  /*replaceLocs=*/false,
                                                  /*replaceTypes=*/true);
            }
          });
        }
      }

      // Clean up the enable fusion calls
      for (auto call : enableFusionFuncs) {
        // Theoretically the user could reference the none return...
        if (call.getResult(0).use_empty())
          call->erase();
      }

      // Locate any function with the elementwise hook and check it is the only
      // call here.

      // The new attributes on the generator.
      SmallVector<NamedAttribute> newAttrs;
      for (NamedAttribute attr : slicedComputeFunction->getAttrs())
        newAttrs.push_back(attr);

      OpBuilder b{ctx};

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
        newAttrs.push_back(NamedAttribute{
            b.getStringAttr("_elementwise_lambda"), asParam.getName()});
      }

      slicedComputeFunction->setAttrs(newAttrs);

      // Shape function.
      mlir::IRMapping mapper;

      // Add the sliced functions to the KGEN as well.
      symTab.insert(slicedComputeFunction);

      // Remove the attributes from the user kernel as that should remain
      // untouched for the user to use directly in their code.
      userKernel.setDecorators(
          KGEN::DecoratorsAttr::get(ctx, kernelMetadata->nonMOGGDecorators));
    }
  }
};

} // namespace
