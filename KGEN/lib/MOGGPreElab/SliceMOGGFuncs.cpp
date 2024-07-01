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
#include "KGEN/MOGGPreElab/MOGGTensorAccessor.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
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
  /// Rewrite a given call to reflect a change in the parameters being passed.
  /// Which parameters are controlled by the caller of this function through the
  /// given lambda.
  void rewriteCallWithNewParams(
      CallOp call, SymbolTable symTab,
      std::function<void(Value, const MOGG::MOGGTensorParamAccessor &,
                         SmallVector<TypedAttr> &, GeneratorOp)>
          updateParams) {
    KGEN::SymbolConstantAttr symbol = call.getCallee();
    FlatSymbolRefAttr flatSym = cast<FlatSymbolRefAttr>(symbol.getSymbol());
    auto calledFunc =
        dyn_cast<KGEN::GeneratorOp>(symTab.lookup(flatSym.getValue()));

    // Could also be a post elaborated function or an ExternalGeneratorOp.
    if (!calledFunc)
      return;

    if (symbol.getParamValues().empty())
      return;

    SmallVector<TypedAttr> newParams;
    for (TypedAttr param : symbol.getParamValues())
      newParams.push_back(param);

    bool shouldUpdate = false;

    // Update the parameters using the caller provided heuristic.
    for (auto [idx, value] : llvm::enumerate(call->getOperands())) {
      std::optional<MOGG::MOGGTensorParamAccessor> callRep =
          getTensorRepFromFunctionInput(calledFunc, idx);
      if (callRep.has_value()) {
        updateParams(value, *callRep, newParams, calledFunc);
        shouldUpdate = true;
      }
    }

    // Obviously don't do anything if we have no values to update.
    if (!shouldUpdate)
      return;

    // Now we have the list of parameters which need to be updated we
    // can rewrite the call to reflect the new lambda.
    auto newSig = calledFunc.getSignature().getSpecializedSignature(
        newParams, [&]() -> mlir::InFlightDiagnostic {
          return calledFunc->emitError(
              "INTERNAL COMPILER ERROR: Parameter specialization "
              "failed: ");
        });
    if (!newSig) {
      signalPassFailure();
      return;
    }

    // Point the call to the new rebinding.
    call.setCalleeAttr(
        KGEN::SymbolConstantAttr::get(flatSym, newParams, newSig));
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
    auto boolFalse = BoolAttr::get(ctx, false);

    // As we rewrite the calls identify the non constant parameters which are
    // being used. We will then replace these too.
    DenseSet<KGEN::ParamDeclRefAttr> ownedMemParams;

    gen.walk([&](CallOp call) {
      auto paramUpdate = [&](Value, const MOGG::MOGGTensorParamAccessor &tensor,
                             SmallVector<TypedAttr> &newParams,
                             GeneratorOp callerGen) {
        if (std::optional<size_t> index = tensor.ownedMemory(callerGen)) {
          if (auto decl = dyn_cast<KGEN::ParamDeclRefAttr>(newParams[*index]))
            ownedMemParams.insert(decl);
          newParams[*index] = boolFalse;
        }
      };

      rewriteCallWithNewParams(call, symTab, paramUpdate);
    });

    mlir::AttrTypeReplacer walker;
    walker.addReplacement(
        [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
          if (ownedMemParams.contains(attr))
            return boolFalse;
          return std::nullopt;
        });

    walker.recursivelyReplaceElementsIn(gen, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);
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
    newAttrs.push_back(NamedAttribute{b.getStringAttr("_in_lambdas"),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames;
    for (const std::string &name : outputLambdaNames)
      outNames.push_back(name);
    newAttrs.push_back(NamedAttribute{b.getStringAttr("_out_lambdas"),
                                      b.getStrArrayAttr(outNames)});
    gen->setAttrs(newAttrs);
  }

  // By checking which tensors have called the `enableFusion` function we can
  // use this information to enable fusion for those which have opted in.
  // Enabling fusion involves materializing a call to the input/output lambda
  // within the body of the function and replacing all previous parameter uses
  // with that value.
  SmallVector<KGEN::ParamDeclareRegionOp>
  enableFusion(GeneratorOp gen, SmallVector<std::string> &inputLambdaNames,
               SmallVector<std::string> &outputLambdaNames,
               SmallVector<KGEN::CallOp> &enableFusionFuncs,
               SymbolTable &symTab) {
    Block *computeBlock = gen.getBody();
    SmallVector<KGEN::ParamDeclareRegionOp> newLambdas;

    for (KGEN::CallOp enableFusionFunc : enableFusionFuncs) {
      std::string newLambdaName;
      Value tensorFusionEnabledOn = enableFusionFunc.getOperand(0);

      bool isInput = true;
      LambdaTemplate *lambda;

      KGEN::ParamDeclRefAttr oldParam;
      if (tensorFusionEnabledOn == gen.getBody()->getArguments().back()) {
        newLambdaName = "output_0_fn";
        outputLambdaNames[0] = newLambdaName;
        lambda = &outLambdaTemplate;
        isInput = false;
        oldParam = dyn_cast<ParamDeclRefAttr>(
            enableFusionFunc.getCallee().getParamValues()
                [MOGG::MOGGTensorParamAccessor::OUTPUT_LAMBDA_IDX]);
      } else {
        lambda = &inLambdaTemplate;
        oldParam = dyn_cast<ParamDeclRefAttr>(
            enableFusionFunc.getCallee().getParamValues()
                [MOGG::MOGGTensorParamAccessor::INPUT_LAMBDA_IDX]);

        // We are dealing with an input.
        for (auto [index, value] :
             llvm::enumerate(gen.getBody()->getArguments())) {
          if (value == tensorFusionEnabledOn) {
            newLambdaName = "input_" + std::to_string(index) + "_fn";
            inputLambdaNames[index] = newLambdaName;
            break;
          }
        }
      }

      // Instead of referring to the `self` argument of the wrapper function
      // which contains the canonical lambda we remap onto the argument of
      // this function which invoked the enable fusion method.
      mlir::IRMapping mapper;

      if (!lambda->templateOp)
        return newLambdas;

      mapper.map(lambda->templateOp.getBody()->getArgument(0),
                 tensorFusionEnabledOn);
      OpBuilder b{computeBlock, computeBlock->begin()};

      // Copy the param region into the body.
      auto newLambda = cast<KGEN::ParamDeclareRegionOp>(
          b.clone(*lambda->canonicalLambda, mapper));

      newLambdas.push_back(newLambda);

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
      // Update all the call sites which use the output tensor to reflect
      // the new lambda.
      for (Operation *user : tensorFusionEnabledOn.getUsers()) {
        if (auto call = dyn_cast<CallOp>(user)) {
          // Don't include any uses within the lambda itself.
          bool inLambda = false;
          Operation *parent = call->getParentOp();
          while (parent) {
            if (parent == newLambda) {
              inLambda = true;
              break;
            }
            parent = parent->getParentOp();
          }

          TypedAttr newParam;
          if (inLambda) {
            // If we are in the lambda then directly remap to point to null.
            auto asVariant =
                cast<KGEN::VariantType>(newLambdaBinding.getType());
            auto lambdaNoneTy = KGEN::VariantAttr::get(
                b.getIntegerAttr(b.getI1Type(), 0), 1, asVariant);
            newParam = lambdaNoneTy;
          } else {
            newParam = newLambdaBinding;
          }

          auto paramUpdate = [&](Value operand,
                                 const MOGG::MOGGTensorParamAccessor &tensor,
                                 SmallVector<TypedAttr> &newParams,
                                 GeneratorOp calledGen) {
            // Update only the tensors fusion has actually been enabled on.
            if (operand != tensorFusionEnabledOn)
              return;

            if (isInput) {
              if (std::optional<size_t> index = tensor.inputLambda(calledGen))
                newParams[*index] = newParam;
            } else {
              if (std::optional<size_t> index = tensor.outputLambda(calledGen))
                newParams[*index] = newParam;
            }
          };
          rewriteCallWithNewParams(call, symTab, paramUpdate);

          // Rewrite all other uses outside calls if this is a concrete
          // parameter reference.
          if (oldParam) {
            paramRebinds[oldParam] = newLambdaBinding;

            gen.walk([&](Operation *op) {
              if (op != newLambda) {
                walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                                    /*replaceLocs=*/false,
                                                    /*replaceTypes=*/true);
              }
            });
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
    return newLambdas;
  }

  void reparameterizeImpl(GeneratorOp gen, SymbolTable &symTab) {

    OpBuilder builder{gen.getContext()};

    auto boolFalse = BoolAttr::get(ctx, false);

    SmallVector<KGEN::ParamDeclAttr> params;
    for (KGEN::ParamDeclAttr p : gen.getInputParams())
      params.push_back(p);

    SmallVector<size_t> paramsToRemove;
    DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> paramsToRewrite;
    DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> paramsToRewriteSigType;

    // Add each tensor parameter as a parameter onto the function.
    for (size_t operand = 0; operand < gen.getNumArguments(); ++operand) {
      std::optional<MOGG::MOGGTensorParamAccessor> tensor =
          getTensorRepFromFunctionInput(gen, operand);

      if (tensor.has_value()) {
        if (std::optional<size_t> index = tensor->ownedMemory(gen)) {
          paramsToRemove.push_back(*index);
          KGEN::ParamDeclRefAttr decl =
              KGEN::ParamDeclRefAttr::get(gen.getInputParams()[*index]);
          paramsToRewrite[decl] = boolFalse;
        }

        // Remove the lambdas from the sig. They will either be fused or none.
        if (std::optional<size_t> index = tensor->inputLambda(gen)) {
          paramsToRemove.push_back(*index);

          KGEN::ParamDeclRefAttr decl =
              KGEN::ParamDeclRefAttr::get(gen.getInputParams()[*index]);
          auto asVariant = cast<KGEN::VariantType>(decl.getType());
          auto lambdaNoneTy = KGEN::VariantAttr::get(
              builder.getIntegerAttr(builder.getI1Type(), 0), 1, asVariant);
          paramsToRewrite[decl] = lambdaNoneTy;
        }

        if (std::optional<size_t> index = tensor->outputLambda(gen)) {
          paramsToRemove.push_back(*index);

          KGEN::ParamDeclRefAttr decl =
              KGEN::ParamDeclRefAttr::get(gen.getInputParams()[*index]);
          auto asVariant = cast<KGEN::VariantType>(decl.getType());
          auto lambdaNoneTy = KGEN::VariantAttr::get(
              builder.getIntegerAttr(builder.getI1Type(), 0), 1, asVariant);
          paramsToRewrite[decl] = lambdaNoneTy;
        }
      }
    }

    // Replace all the parameters with their `none` equivalent. At this point
    // all enabled fusion should already have been dealt with in the enable
    // fusion funcion.
    mlir::AttrTypeReplacer walker;
    walker.addReplacement(
        [&](KGEN::ParamDeclRefAttr attr) -> std::optional<TypedAttr> {
          auto itr = paramsToRewrite.find(attr);
          if (itr != paramsToRewrite.end())
            return itr->second;
          return {};
        });

    gen.walk([&](Operation *op) {
      walker.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);
    });

    // Specialize the generator using the above parameters.
    SmallVector<TypedAttr> paramsToSpecialize;

    // Remove any parameters which we know know are unused.
    std::sort(paramsToRemove.begin(), paramsToRemove.end(),
              std::greater<size_t>());
    for (size_t i : paramsToRemove)
      params.erase(params.begin() + i);

    // Update the sig to partially specialize on those function types.
    SignatureType oldSig = gen.getSignature();
    gen.setSignature(SignatureType::remapToSignature(
        params, {}, gen.getFunctionType(),
        /*argConventions=*/oldSig.getArgConventions(),
        /*fnEffects=*/oldSig.getFnEffects(),
        /*metadata*/ oldSig.getMetadata(), [&] {
          return gen->emitError("Failed to remap generator signature.");
        }));

    // Remove the old params from the function.
    gen.setInputParams(params);

    // We still need to brute force any is_owned to true, looking through the
    // params isn't enough as it may be defaulted.
    markAllTensorsAsOwned(gen, symTab);
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
      std::string name =
          (Twine(userKernel.getSymName()) + Twine("_COMPUTE")).str();
      slicedComputeFunction.setSymName(name);

      // Search for any function which allocates a new tensor and a move from
      // that into one of the input operands (meaning it is actually an output).
      KGEN::CallOp allocationFunc, constructor;

      // If the user has any call to enable fusion then we turn on fusion for
      // that tensor.
      SmallVector<KGEN::CallOp> enableFusionFuncs;
      SmallVector<KGEN::CallOp> deconstructors;

      // Scan the kernel and identify the callsites of annotated functions that
      // we can understand.
      for (KGEN::CallOp call : slicedComputeFunction.getOps<KGEN::CallOp>()) {
        auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (!func)
          continue;

        if (func->hasAttr(Decorators::TENSOR_ALLOC.attr))
          allocationFunc = call;
        else if (func->hasAttr(Decorators::TENSOR_COPY.attr))
          constructor = call;
        else if (func->hasAttr(Decorators::ENABLE_FUSION.attr))
          enableFusionFuncs.push_back(call);
        else if (func->hasAttr(Decorators::TENSOR_DECONSTRUCT.attr))
          deconstructors.push_back(call);
      }

      // Strip all debug info. Its too annoying to maintain and there is no way
      // to actually debug the sliced kernel directly. Users would debug the
      // base kernel.
      slicedComputeFunction.walk([](Operation *op) {
        if (llvm::isa_and_nonnull<DebugInfo::DebugInfoDialect>(
                op->getDialect()))
          op->erase();
      });

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
      assert(slicedComputeFunction.getSignature().hasMemoryOnlyResult());
      Value outputTensor =
          slicedComputeFunction.getBody()->getArguments().back();

      SmallVector<KGEN::ParamDeclareRegionOp> addedLambdas;

      bool isView = false;
      // Any MOGG annotated kernel which has no allocation should be treated as
      // a view.
      if (!allocationFunc) {
        isView = true;
      } else if (!constructor) {
        // Exit and clean up if the kernel is not what we expect. Allocators and
        // constructors are expected to appear as a pair.
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

        // Turn on fusion for any tensors which have set fusion.
        addedLambdas =
            enableFusion(slicedComputeFunction, inputLambdaNames,
                         outputLambdaNames, enableFusionFuncs, symTab);
      }

      // Strip all debug info. Its too annoying to maintain and there is no way
      // to actually debug the sliced kernel directly. Users would debug the
      // base kernel.
      slicedComputeFunction.walk([](Operation *op) {
        if (llvm::isa_and_nonnull<DebugInfo::DebugInfoDialect>(
                op->getDialect()))
          op->erase();
      });

      // Add compute function part to the module, i.e the kernel sans
      // allocation.
      symTab.insert(slicedComputeFunction);

      // Remove all tensor parameters which have been instantiated explicitly by
      // MOGG, namely the input / output lambdas and the owned memory bool
      // attribute.
      reparameterizeImpl(slicedComputeFunction, symTab);

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
