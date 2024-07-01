//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/MOGGTensorAccessor.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "Helpers.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_OUTLINEMOGGFUNCS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {
class OutlineMOGGFuncsPass
    : public M::KGEN::MOGGPreElab::impl::OutlineMOGGFuncsBase<
          OutlineMOGGFuncsPass> {
private:
  MLIRContext *ctx;

private:
  void outlineFunction(GeneratorOp gen,
                       SmallVector<KGEN::ParamDeclareRegionOp> &lambdas,
                       CallOp elementwiseOp, SymbolTable &symTab) {
    // We are either outlining the full function or just the inner elementwise.
    SmallVector<Operation *> opsToClone;
    ArrayRef<Type> returnTypes;

    // If this is elementwise outline just the elementwise.
    if (elementwiseOp) {
      auto elemwiseLambda = elementwiseOp.getParamValues().back();
      auto asParam = dyn_cast<ParamDeclRefAttr>(elemwiseLambda);

      // Look for the lambda and clone the body.
      for (auto lambda : gen.getOps<KGEN::ParamDeclareRegionOp>()) {
        if (lambda.getParamDecl().getName() == asParam.getName()) {
          for (Operation &op : lambda.getOps())
            opsToClone.push_back(&op);
          returnTypes = lambda.getFunctionType().getResults();
        }
      }
    } else {
      for (Operation &op : gen.getOps()) {
        // Don't include the input / output lambdas in the cloning. These are
        // the input interfaces on the kernel we will clone into MOGG.
        if (std::any_of(lambdas.begin(), lambdas.end(),
                        [&](KGEN::ParamDeclareRegionOp lambda) {
                          return &op == lambda;
                        }))
          continue;
        opsToClone.push_back(&op);
      }
      returnTypes = gen.getFunctionType().getResults();
    }

    if (opsToClone.size() == 0)
      return;

    DenseSet<StringAttr> definedParams;

    // Use set vectors for deterministic traversal. Identify parameters used in
    // the block and any uses which originate from above.
    llvm::SetVector<KGEN::ParamDeclRefAttr> usedParams;
    DenseSet<Value> valuesDefinedInBlock;
    llvm::SetVector<Value> usesFromAbove;

    // Add a given decl to the list of param decls we know about and are
    // tracking. Removing from the seen parameters if needed.
    auto trackDefinedParams = [&](KGEN::ParamDeclAttr decl) {
      usedParams.remove(KGEN::ParamDeclRefAttr::get(decl));
      definedParams.insert(decl.getName());
    };

    mlir::AttrTypeWalker walker;
    walker.addWalk([&](KGEN::ParamDeclRefAttr ref) {
      if (!definedParams.contains(ref.getName()))
        usedParams.insert(ref);
    });

    // Walk the ops and identify all parameters or SSA value inputs to the block
    // which will become inputs to our outlined function.
    auto identifyInputParamsAndValues = [&](Operation *op) {
      if (op == gen)
        return mlir::WalkResult::advance();

      for (Value v : op->getResults()) {
        valuesDefinedInBlock.insert(v);
        usesFromAbove.remove(v);
      }

      for (Region &region : op->getRegions()) {
        for (Value v : region.getArguments()) {
          valuesDefinedInBlock.insert(v);
          usesFromAbove.remove(v);
        }

        for (Block &block : region.getBlocks()) {
          for (Value v : block.getArguments()) {
            valuesDefinedInBlock.insert(v);
            usesFromAbove.remove(v);
          }
        }
      }

      for (Value v : op->getOperands()) {
        if (!valuesDefinedInBlock.contains(v))
          usesFromAbove.insert(v);
      }

      for (Type t : op->getOperandTypes())
        walker.walk(t);
      for (Type t : op->getResultTypes())
        walker.walk(t);
      walker.walk(op->getAttrDictionary());

      // Track parameters within lambda blocks ect.
      if (auto definesParam = dyn_cast<KGEN::DeclInterface>(op)) {
        for (KGEN::ParamDeclAttr decl : definesParam.getInputParams())
          trackDefinedParams(decl);
        for (KGEN::ParamDeclAttr decl : definesParam.getResultParams())
          trackDefinedParams(decl);
      }

      if (auto definesParam = dyn_cast<KGEN::ParamOpInterface>(op)) {
        // Remove any parameters which are defined internally within our region.
        definesParam.walkDeclarations(
            [&](KGEN::ParamDeclAttr decl) { trackDefinedParams(decl); });
        definesParam.walkDefinitions(
            [&](KGEN::ParamDeclAttr decl, const KGEN::ParamDefValue &) {
              trackDefinedParams(decl);
            });
      }

      return mlir::WalkResult::advance();
    };

    for (Operation *op : opsToClone)
      op->walk(identifyInputParamsAndValues);

    // Translate the input params / values into types needed to build the
    // signature.
    SmallVector<Type> inputArgTypes;
    SmallVector<KGEN::ParamDeclAttr> asDecls;
    SmallVector<KGEN::ParamDeclRefAttr> paramsAsArgument;
    SmallVector<TypedAttr> paramArgs;

    for (auto p : usedParams) {
      asDecls.push_back(KGEN::ParamDeclAttr::get(p));
      paramArgs.push_back(p);
    }

    inputArgTypes.reserve(usesFromAbove.size());
    for (Value v : usesFromAbove)
      inputArgTypes.push_back(v.getType());

    OpBuilder builder{ctx};
    builder.setInsertionPoint(gen);

    // Create the outlined function to call.
    auto newFuncType = FunctionType::get(ctx, inputArgTypes, returnTypes);
    auto sigType = SignatureType::remapToSignature(
        asDecls, {}, newFuncType,
        /*argConventions=*/{}, KGEN::impl::FnEffects::Capturing);

    std::string name = (Twine(gen.getSymName()) + Twine("_OUTLINED")).str();
    auto outlinedFunction = builder.create<KGEN::GeneratorOp>(
        gen.getLoc(), builder.getStringAttr(name), sigType, newFuncType,
        asDecls, ArrayRef<KGEN::ParamDeclAttr>{});

    // We are inlining the function we just outlined because the purpose of the
    // outlining is just to make sure the graph compiler works on a minimal set
    // of changes
    outlinedFunction.setInlineLevel(KGEN::InlineLevel::Always);

    Block &block = outlinedFunction.getCallableRegion()->emplaceBlock();
    builder.setInsertionPointToStart(&block);

    IRMapping mapper;

    // Pass all the original arguments to the kernel.
    for (Value v : usesFromAbove) {
      Value newArg = block.addArgument(v.getType(), v.getLoc());
      mapper.map(v, newArg);
    }

    for (Operation *op : opsToClone)
      builder.clone(*op, mapper);

    // Get the block we're adding before we remove it.
    Block *insertPt = opsToClone[0]->getBlock();

    // Remove the now dead ops.
    for (Operation *op : opsToClone) {
      op->dropAllUses();
      op->erase();
    }

    builder.setInsertionPointToEnd(insertPt);

    // Update the insertion point so we add the call in the right place.
    symTab.insert(outlinedFunction);

    // Finally add the call to the inlined function.
    auto flatSym = FlatSymbolRefAttr::get(ctx, outlinedFunction.getNameAttr());

    auto specializedSig =
        outlinedFunction.getSignature().getSpecializedSignature(
            paramArgs, outlinedFunction.getLoc());
    auto symbol =
        KGEN::SymbolConstantAttr::get(flatSym, paramArgs, specializedSig);

    // Create the KGEN parameter bindings. I.E the <> "template" parameters.
    // Note this is empty as we expect all parameters to be bound in the above
    // sig.
    auto callOp = builder.create<KGEN::CallOp>(
        outlinedFunction.getLoc(), symbol.getType().getResults(), symbol,
        usesFromAbove.getArrayRef());

    builder.create<KGEN::ReturnOp>(gen.getLoc(), callOp->getResults());
  }

  // An elementwise op is an op which is purely represented by a call to the
  // elementwise generator. I.E there are no other ops left after slicing.
  CallOp checkKernelIsPureElementwise(CallOp elementwiseOp,
                                      KGEN::GeneratorOp gen) {
    if (!elementwiseOp)
      return elementwiseOp;

    auto opHasSideEffect = [=](Operation *op) {
      return (!isa<POP::StackAllocationOp>(op) && !isPure(op)) ||
             op->hasTrait<OpTrait::IsTerminator>();
    };

    // Apply trivial DCE on top level ops to remove now unused operations.
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : llvm::make_early_inc_range(gen.getOps())) {
        if (opHasSideEffect(&op) || op.hasTrait<OpTrait::IsTerminator>())
          continue;

        if (op.use_empty()) {
          op.erase();
          changed = true;
        }
      }
    }

    for (Operation &op : gen.getOps()) {
      if (&op == elementwiseOp)
        continue;
      if (isa<KGEN::ParamDeclareOp, KGEN::ParamDeclareRegionOp, KGEN::ReturnOp,
              KGEN::ParamConstantOp>(op))
        continue;
      // Should not be marked elementwise.
      return nullptr;
    }
    return elementwiseOp;
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ctx = mod.getContext();
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symTab = analysis.getTopLevelSymbolTable();

    for (GeneratorOp kernel : mod.getOps<GeneratorOp>()) {
      // Skip non-kernels.
      if (!(kernel->hasAttr(SLICED_ATTR) || kernel->hasAttr(ALLOCS_ATTR)))
        continue;

      // Pull the lambda names off the kernel so we can find their decl in the
      // implementation to avoid outlining them.
      llvm::StringSet<> lambdas;
      if (auto inLambdaAttr = kernel->getAttrOfType<ArrayAttr>("_in_lambdas")) {
        for (StringRef name : inLambdaAttr.getAsValueRange<StringAttr>())
          lambdas.insert(name);
      }
      if (auto outLambdaAttr =
              kernel->getAttrOfType<ArrayAttr>("_out_lambdas")) {
        for (StringRef name : outLambdaAttr.getAsValueRange<StringAttr>())
          lambdas.insert(name);
      }

      // Identify the input / output lambdas which have been added.
      SmallVector<KGEN::ParamDeclareRegionOp> addedLambdas;
      for (auto lambda : kernel.getOps<KGEN::ParamDeclareRegionOp>()) {
        if (lambdas.contains(lambda.getParamDecl().getName()))
          addedLambdas.push_back(lambda);
      }

      // If this is an elementwise kernel we are expecting to see a call to the
      // elementwise generator.
      KGEN::CallOp elementwiseOp = nullptr;

      // Views should never be marked as elementwise even if they called it.
      if (!kernel->hasAttr("_view")) {
        // Search for the elementwise kernel.
        for (auto call : kernel.getOps<KGEN::CallOp>()) {
          auto func = dyn_cast_or_null<KGEN::GeneratorOp>(symTab.lookup(
              cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));

          // Allowed to fail as it could be a call to an ExternalGenerator
          if (!func)
            continue;

          if (func->hasAttr(Decorators::ELEM_HOOK.attr))
            elementwiseOp = call;
        }

        // Ensure elementwise kernels are actually elementwise.
        elementwiseOp = checkKernelIsPureElementwise(elementwiseOp, kernel);
      }

      // Outline the actual work of the function.
      // 1. If it is elementwise, outline the body of the elementwise lambda
      // 2. Otherwise outline everything other than the lambdas.
      outlineFunction(kernel, addedLambdas, elementwiseOp, symTab);

      // Tell MOGG this thing is elementwise.
      if (elementwiseOp) {
        // Last parameter is known to be the lambda...
        auto elemwiseLambda = elementwiseOp.getParamValues().back();
        auto asParam = dyn_cast<ParamDeclRefAttr>(elemwiseLambda);

        // The new attributes on the generator.
        SmallVector<NamedAttribute> attrsToAdd;

        // Add all the old attributes.
        for (NamedAttribute attr : kernel->getAttrs())
          attrsToAdd.push_back(attr);

        OpBuilder builder{ctx};
        attrsToAdd.push_back(NamedAttribute{
            builder.getStringAttr("_elementwise_lambda"), asParam.getName()});
        kernel->setAttrs(attrsToAdd);
      }
    }
  }
};
} // namespace
