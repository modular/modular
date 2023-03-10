//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "outline-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_OUTLINECLOSURES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct OutlineClosuresPass
    : M::KGEN::impl::OutlineClosuresBase<OutlineClosuresPass> {
  void runOnOperation() override;
};
} // namespace

void OutlineClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  // Walk over all the param.declare.region ops and create structs with the SSA
  // captures, use bind_signature to deal with parameter captures.
  OpBuilder b(theModule);
  b.setInsertionPointToStart(theModule.getBody());

  unsigned counter = 0, varCounter = 0;
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    // Calculate the parameter decls and uses for the region decl's parent.
    ParameterUseDefGraph uses(generator.getBodyRegion());
    uses.calculate(paramCache);

    // We'll use this a lot here - pull it out into a little lambda.
    auto getUniqueName = [&](const Twine &suffix) {
      return b.getStringAttr(getUniqueSymbolName(
          (generator.getName() + suffix).str(), symtab, counter));
    };

    auto getUniqueVarName = [&](StringRef suffix) {
      return b.getStringAttr(generator.getName() + suffix + "_" +
                             Twine(varCounter++));
    };

    bool hadError = false;
    generator.walk([&](ParamDeclareRegionOp regionDecl) {
      LLVM_DEBUG(llvm::dbgs()
                 << "//===-----\nLifting closure: " << regionDecl << "\n");
      StringRef regionName = regionDecl.getParamDecl().getName();

      // Value captures are easy (ish)
      SmallVector<Value> captures;
      bool isolated = M::operationIsIsolatedFromAbove(regionDecl, &captures);

      // If the body is not isolated from above *and* it's not marked
      // always_inline, emit an error.
      if (!isolated &&
          regionDecl.getAlwaysInlineLevel() == AlwaysInlineLevel::Disabled) {
        regionDecl.emitError(
            "non-isolated region must be marked always_inline");
        hadError = true;
        return;
      }

      LLVM_DEBUG(llvm::dbgs() << "Found value captures: [";
                 llvm::interleaveComma(captures, llvm::dbgs());
                 llvm::dbgs() << "]\n");

      // Create a struct with the correct parameter decls if needed (i.e. if
      // there are any captures).
      StringAttr globalVar = nullptr;
      POP::StructType structType = nullptr;
      if (!isolated) {
        structType = b.getType<POP::StructType>(map_to_vector(
            captures, [](Value capture) { return capture.getType(); }));

        LLVM_DEBUG(llvm::dbgs()
                   << "Created capture struct: " << structType << "\n");

        // 'Create' a global variable (really just a StringAttr).
        globalVar = getUniqueVarName("_context_var");
      }

      // Collect any parameters used from above that we need to capture for the
      // lifted generator.
      llvm::SetVector<ParamDeclAttr> necessaryDecls;
      SmallVector<ParamDeclRefAttr> capturedParamValues;
      auto regionDeclUses = uses.nestedScopes.find(&regionDecl.getBodyRegion());
      assert(regionDeclUses != uses.nestedScopes.end());

      // Scan the captured values for captured parameters.
      ParameterCollector collector(paramCache);
      SmallVector<ParamDeclRefAttr, 16> capturedUses;
      for (Value capture : captures) {
        capturedUses.clear();
        bool unused;
        collector.collectUsesFromType(capture.getType(), capturedUses, unused);
        for (ParamDeclRefAttr capturedUse : capturedUses) {
          Operation *declOp =
              regionDeclUses->second.decls.find(capturedUse.getName())
                  ->second.declOp;
          if (!regionDecl->isAncestor(declOp))
            regionDeclUses->second.usesFromAbove.insert(capturedUse);
        }
      }

      for (ParamDeclRefAttr useFromAbove :
           regionDeclUses->second.usesFromAbove) {
        auto decl =
            ParamDeclAttr::get(useFromAbove.getName(), useFromAbove.getType());
        if (necessaryDecls.insert(decl))
          capturedParamValues.push_back(useFromAbove);
      }

      LLVM_DEBUG(llvm::dbgs() << "Found parameter captures: [";
                 llvm::interleaveComma(necessaryDecls, llvm::dbgs());
                 llvm::dbgs() << "]\n");

      SignatureType bodySignature = regionDecl.getFullSignature();

      // The value signature is pretty simple here, just captures and then any
      // original arguments.
      SmallVector<Value> liftedInputs = captures;
      llvm::append_range(liftedInputs,
                         regionDecl.getBodyRegion().getArguments());
      LLVM_DEBUG(llvm::dbgs() << "Lifted region will take inputs: [\n\t";
                 llvm::interleave(liftedInputs, llvm::dbgs(), ",\n\t");
                 llvm::dbgs() << "\n]\n");
      auto liftedValueSignature =
          FunctionType::get(&getContext(), ValueRange(liftedInputs).getTypes(),
                            bodySignature.getValueResults());

      // The parameter signature is just the necessary decls + original
      // arguments, and then any of the original results.
      for (ParamDeclAttr inputParam : regionDecl.getInputParams()) {
        bool inserted = necessaryDecls.insert(inputParam);
        assert(inserted && "nested parameter declaration was duplicated?");
      }

      // The lifted generator needs to be always_inline, so we add that to the
      // FnEffects.
      auto liftedSignature = SignatureType::get(
          b.getAttr<ParamDeclArrayAttr>(necessaryDecls.getArrayRef()),
          bodySignature.getResultParams(), liftedValueSignature,
          b.getAttr<MetadataAttr>(liftedValueSignature.getNumInputs()));

      // Now lift the body out into its own generator.
      b.setInsertionPoint(generator);
      auto lifted = b.create<GeneratorOp>(
          regionDecl.getLoc(), getUniqueName("_" + regionName),
          TypeAttr::get(liftedSignature),
          b.getAttr<ConstraintArrayAttr>(ArrayRef<ConstraintAttr>{}),
          b.getAttr<AlwaysInlineLevelAttr>(
              regionDecl.getAlwaysInlineLevel() == AlwaysInlineLevel::Disabled
                  ? AlwaysInlineLevel::Enabled
                  : regionDecl.getAlwaysInlineLevel()));
      symtab.insert(lifted);
      auto liftedSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(lifted.getSymNameAttr()), liftedSignature);

      // Create the generator's body.
      if (!isolated) {
        // Not isolated, so we have to clone the ops in so we can remap
        // arguments.
        auto *newBody = new Block;
        IRMapping map;
        // Handle the captures first.
        for (Value capture : captures)
          map.map(capture,
                  newBody->addArgument(capture.getType(), capture.getLoc()));

        // Then handle the original SSA arguments.
        for (Value prevArg : regionDecl.getArguments())
          map.map(prevArg,
                  newBody->addArgument(prevArg.getType(), prevArg.getLoc()));

        b.setInsertionPointToStart(newBody);
        for (Operation &op : regionDecl.getOps())
          b.clone(op, map);

        lifted.getBodyRegion().push_back(newBody);
      } else {
        // Take the body from the param region.
        lifted.getBodyRegion().takeBody(regionDecl.getBodyRegion());
      }
      LLVM_DEBUG(llvm::dbgs() << "Created lifted region: " << lifted << "\n");

      LLVM_DEBUG({
        if (failed(mlir::verify(lifted)))
          return signalPassFailure();
      });

      // Create a wrapper that knows how to handle the global variable. It has
      // the same parameter signature as the lifted region, but it has the same
      // value signature as the original parameter region (no captures - those
      // come from global variables).
      auto wrapperSignature = SignatureType::get(
          liftedSignature.getInputParams(), liftedSignature.getResultParams(),
          bodySignature.getValues(), bodySignature.getMetadata());

      b.setInsertionPoint(generator);
      auto liftedWrapper = b.create<GeneratorOp>(
          regionDecl.getLoc(), getUniqueName("_" + regionName + "_wrapper"),
          TypeAttr::get(wrapperSignature),
          b.getAttr<ConstraintArrayAttr>(ArrayRef<ConstraintAttr>{}),
          regionDecl.getAlwaysInlineLevelAttr());
      symtab.insert(liftedWrapper);
      auto wrapperSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(liftedWrapper.getNameAttr()), wrapperSignature);

      // Fill the body of the wrapper.
      liftedWrapper.getBodyRegion().push_back(new Block);
      b.setInsertionPointToStart(liftedWrapper.getBody());
      SmallVector<Value> callArgs;
      if (!isolated) {
        assert(globalVar && structType &&
               "global variable name/type was undefined?");
        auto load = b.create<POP::CompilerGlobalLoadOp>(regionDecl.getLoc(),
                                                        structType, globalVar);
        // Create accesses for each capture.
        for (size_t i = 0, e = structType.getNumElements(); i < e; ++i) {
          callArgs.push_back(
              b.create<POP::StructExtractOp>(load.getLoc(), load, i));
        }
      }

      // Add the original arguments to the call after the captures. Since the
      // captures are the first N arguments, we can simply drop them.
      for (BlockArgument liftedArg :
           llvm::drop_begin(lifted.getArguments(), captures.size())) {
        callArgs.push_back(liftedWrapper.getBodyRegion().addArgument(
            liftedArg.getType(), liftedArg.getLoc()));
      }

      // Create result parameter decls from the lifted region, and get decl refs
      // for the actual ReturnOp.
      SmallVector<ParamDeclAttr> resultDecls;
      SmallVector<TypedAttr> returnRefs;
      for (auto [idx, resultParam] :
           llvm::enumerate(lifted.getResultParams())) {
        auto declName = b.getStringAttr("__resultParam_" + Twine(idx));
        // If something is somehow named __resultParam_0 then just increment the
        // counter till it works.
        while (llvm::find_if(lifted.getInputParams(), [&](ParamDeclAttr decl) {
                 return decl.getName() == declName;
               }) != lifted.getInputParams().end())
          declName = b.getStringAttr("__resultParam_" + Twine(++idx));

        resultDecls.push_back(
            ParamDeclAttr::get(declName, resultParam.getType()));
        returnRefs.push_back(
            ParamDeclRefAttr::get(declName, resultParam.getType()));
      }

      // We need to set the parameter bindings for the call to the lifted
      // region. This basically just means binding the wrapper's input params to
      // a ref.
      SmallVector<ParamBindAttr> symbolBindings;
      for (ParamDeclAttr decl : liftedWrapper.getInputParams()) {
        symbolBindings.push_back(
            ParamBindAttr::get(decl.getName(), ParamDeclRefAttr::get(decl)));
      }
      LLVM_DEBUG(llvm::dbgs() << "Bindings: [\n\t";
                 llvm::interleave(symbolBindings, llvm::dbgs(), ",\n\t");
                 llvm::dbgs() << "\n]");

      // Create the specialized call to the lifted region.
      auto liftedCallSymbol = SymbolConstantAttr::get(
          liftedSymbol.getSymbol(),
          ParamBindArrayAttr::get(&getContext(), symbolBindings),
          liftedSymbol.getType().getSpecializedSignature(symbolBindings, [&]() {
            return regionDecl->emitError(
                "could not specialize the lifted generator's signature: ");
          }));
      auto callResults = b.create<CallOp>(
          regionDecl.getLoc(), lifted.getSignature().getValueResults(),
          liftedCallSymbol, ParamDeclArrayAttr::get(&getContext(), resultDecls),
          callArgs);
      if (!returnRefs.empty())
        b.create<ParamResultBindOp>(regionDecl.getLoc(), returnRefs);
      b.create<ReturnOp>(regionDecl.getLoc(), callResults.getResults());

      LLVM_DEBUG(llvm::dbgs()
                 << "Created lifted region wrapper: " << liftedWrapper << "\n");

      LLVM_DEBUG({
        if (failed(mlir::verify(liftedWrapper)))
          return signalPassFailure();
      });

      Attribute bindSignature = wrapperSymbol;
      // If we have parameter captures, create a bind_signature operator.
      if (necessaryDecls.size() != regionDecl.getInputParams().size()) {
        // OK cool, now we need a partial binding. First we insert the lifted
        // symbol at the beginning of the vector.
        SmallVector<TypedAttr> partialBindings = {wrapperSymbol};
        llvm::append_range(partialBindings, capturedParamValues);
        for (ParamDeclAttr decl : regionDecl.getInputParams())
          partialBindings.push_back(UnboundAttr::get(decl.getType()));
        LLVM_DEBUG(llvm::dbgs() << "Partial bindings: [\n\t";
                   llvm::interleave(partialBindings, llvm::dbgs(), ",\n\t");
                   llvm::dbgs() << "\n]\n");
        bindSignature =
            ParamOperatorAttr::get(POC::BindSignature, partialBindings);
      }

      // Now replace the region decl with a partial binding to the lifted
      // wrapper.
      // Create a container for the struct with all the various captures.
      if (!isolated) {
        // We have to find the earliest possible insertion point, so we start
        // from the beginning of the generator itself.
        b.setInsertionPointToStart(generator.getBody());
        OpBuilder::InsertPoint insertPt = b.saveInsertionPoint();
        // Helper to update the insertion point given a Value.
        auto updateInsertPt = [&](Value val) {
          if (auto blockArg = dyn_cast<BlockArgument>(val))
            b.setInsertionPointToStart(blockArg.getOwner());
          else
            b.setInsertionPointAfter(val.getDefiningOp());
          insertPt = b.saveInsertionPoint();
        };

        for (Value c : captures) {
          // If the capture properly dominates the insert point, then we are
          // fine.
          if (domInfo.properlyDominates(c, &*insertPt.getPoint()))
            continue;

          // Otherwise, we need to reset the insert point.
          updateInsertPt(c);
        }

        b.restoreInsertionPoint(insertPt);

        assert(globalVar && structType &&
               "global variable name/type/struct was undefined?");

        auto container = b.create<POP::StructConstructOp>(regionDecl.getLoc(),
                                                          structType, captures);

        // Get a pointer to the global and store the container in it.
        b.create<POP::CompilerGlobalStoreOp>(regionDecl.getLoc(), globalVar,
                                             container);
      }

      // Set the insertion point to the regionDecl for the parameter
      // declaration.
      b.setInsertionPoint(regionDecl);

      // Create the decl that replaces the regionDecl with its parameter being
      // this new partial binding.
      b.create<ParamDeclareOp>(regionDecl.getLoc(), regionDecl.getParamDecl(),
                               bindSignature);

      // And we can drop the regionDecl now, we're done with it.
      regionDecl->erase();
    });
    if (hadError)
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Modified generator: " << generator << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "Finished outlining closures\n");
}
