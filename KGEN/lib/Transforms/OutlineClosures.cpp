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
#include "mlir/Transforms/RegionUtils.h"
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
      llvm::SetVector<Value> captures;
      mlir::getUsedValuesDefinedAbove(regionDecl->getRegions(), captures);
      bool isolated = captures.empty();
      if (!isolated && !regionDecl.getSignature().isCapturing()) {
        InFlightDiagnostic diag = mlir::emitError(regionDecl.getLoc())
                                  << "nested function is marked as "
                                     "@noncapturing, but it captures values";
        Value capture = captures.front();
        Operation *user =
            *llvm::find_if(capture.getUsers(), [&](Operation *op) {
              return regionDecl->isProperAncestor(op);
            });
        diag.attachNote(user->getLoc()) << "use of captured value here";
        diag.attachNote(capture.getLoc()) << "captured value defined here";
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
      llvm::SetVector<ParamDeclAttr> capturedParamDecls;
      SmallVector<ParamDeclRefAttr> capturedParamValues;
      Region &region = regionDecl.getBodyRegion();
      auto regionDeclUses = uses.nestedScopes.find(&region);
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
        if (capturedParamDecls.insert(decl))
          capturedParamValues.push_back(useFromAbove);
      }

      LLVM_DEBUG(llvm::dbgs() << "Found parameter captures: [";
                 llvm::interleaveComma(capturedParamDecls, llvm::dbgs());
                 llvm::dbgs() << "]\n");

      // Create a wrapper that knows how to handle the global variable. It has
      // the same parameter signature as the lifted region, but it has the same
      // value signature as the original parameter region (no captures - those
      // come from global variables).
      SmallVector<ParamDeclAttr> inputParamDecls(
          capturedParamDecls.getArrayRef());
      llvm::append_range(inputParamDecls, regionDecl.getInputParams());

      SignatureType wrapperSignature = IndexRefRemapper::prependParams(
          regionDecl.getSignature(), capturedParamDecls.getArrayRef());

      b.setInsertionPoint(generator);
      auto liftedWrapper = b.create<GeneratorOp>(
          regionDecl.getLoc(), getUniqueName("_" + regionName),
          TypeAttr::get(wrapperSignature), regionDecl.getFunctionTypeAttr(),
          ParamDeclArrayAttr::get(&getContext(), inputParamDecls),
          regionDecl.getResultParamsAttr(),
          ConstraintArrayAttr::get(&getContext(), {}),
          regionDecl.getAlwaysInlineLevelAttr());
      symtab.insert(liftedWrapper);
      auto wrapperSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(liftedWrapper.getNameAttr()), wrapperSignature);

      // Take the body from the param region.
      liftedWrapper.getBodyRegion().takeBody(region);

      // Add the original arguments to the call after the captures. Since the
      // captures are the last N arguments, we can simply drop them.
      b.setInsertionPointToStart(liftedWrapper.getBody());

      // Fill the body of the wrapper.
      if (!isolated) {
        assert(globalVar && structType &&
               "global variable name/type was undefined?");
        auto load = b.create<POP::CompilerGlobalLoadOp>(regionDecl.getLoc(),
                                                        structType, globalVar);
        // Create accesses for each capture.
        for (auto [idx, capture] : llvm::enumerate(captures)) {
          mlir::replaceAllUsesInRegionWith(
              capture, b.create<POP::StructExtractOp>(load.getLoc(), load, idx),
              liftedWrapper.getBodyRegion());
        }
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

      LLVM_DEBUG(llvm::dbgs()
                 << "Created lifted region wrapper: " << liftedWrapper << "\n");

      LLVM_DEBUG({
        if (failed(mlir::verify(liftedWrapper)))
          return signalPassFailure();
      });

      Attribute bindSignature = wrapperSymbol;
      // If we have parameter captures, create a bind_signature operator.
      if (!capturedParamValues.empty()) {
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

        auto container = b.create<POP::StructCreateOp>(
            regionDecl.getLoc(), structType, llvm::to_vector(captures));

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
                               cast<TypedAttr>(bindSignature));

      // And we can drop the regionDecl now, we're done with it.
      regionDecl->erase();
    });
    if (hadError)
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Modified generator: " << generator << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "Finished outlining closures\n");
}
