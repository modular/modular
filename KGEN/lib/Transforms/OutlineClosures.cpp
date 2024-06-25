//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
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
  using OutlineClosuresBase::OutlineClosuresBase;

  void runOnOperation() override;
};
} // namespace

/// Reconstruct the signature using a list of named input parameters and indices
/// indicating which one of them are variadic. These parameters are prepended to
/// the current signature and references are remapped to index references.
static SignatureType prependParams(SignatureType sig,
                                   ArrayRef<ParamDeclAttr> parentParams) {
  assert(!sig.getMetadata() && "unlowered lit signature");

  IndexRefRemapper remapper(parentParams, /*resultParams=*/{},
                            parentParams.size());
  SmallVector<Type> inputParamTypes;
  for (ParamDeclAttr param : parentParams)
    inputParamTypes.push_back(remapper.replace(param.getType()));
  for (Type type : sig.getInputParamTypes())
    inputParamTypes.push_back(remapper.replace(type));

  return SignatureType::get(remapper.replace(sig.getValues()), inputParamTypes,
                            remapper.replace(sig.getResultParamTypes()),
                            sig.getArgConventions(), sig.getFnEffects());
}

void OutlineClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  // Walk over all the param.declare.region ops and create structs with the SSA
  // captures, use bind_signature to deal with parameter captures.
  unsigned counter = 0;
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    unsigned varCounter = 0;

    // Calculate the parameter decls and uses for the region decl's parent.
    ParameterUseDefGraph uses(generator.getBodyRegion());
    uses.calculate(paramCache);

    bool hadError = false;
    SmallVector<Operation *> toErase;
    generator.walk([&](ParamDeclareRegionOp regionDecl) {
      StringRef regionName = regionDecl.getParamDecl().getName();

      // Value captures are easy (ish)
      llvm::SetVector<Value> captures;
      mlir::getUsedValuesDefinedAbove(regionDecl->getRegions(), captures);
      if (!captures.empty() && !regionDecl.getSignature().isCapturing()) {
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

      // We will use this builder to build the lifted generator.
      ImplicitLocOpBuilder b(regionDecl->getLoc(), regionDecl.getContext());

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
        bool unused;
        {
          CompilerTimeTraceScope traceScope("collectParameters");
          collector.collectUsesFromType(capture.getType(), capturedUses,
                                        unused);
        }
      }

      // Scan locations for captured parameters when in a debug build.
      if (debugBuild) {
        regionDecl.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
          // Since nested regions aren't being deleted, walk over them.
          if (op != regionDecl && isa<ParamDeclareRegionOp>(op))
            return WalkResult::skip();
          bool unused;
          collector.collectUsesFromAttr(op->getLoc(), capturedUses, unused);
          return WalkResult::advance();
        });
      }

      // Add additional uses to the captures set.
      for (ParamDeclRefAttr capturedUse : capturedUses) {
        auto declOpIter =
            regionDeclUses->second.decls.find(capturedUse.getName());
        if (declOpIter == regionDeclUses->second.decls.end())
          continue;
        Operation *declOp = declOpIter->second.declOp;
        if (!regionDecl->isAncestor(declOp))
          regionDeclUses->second.usesFromAbove.insert(capturedUse);
      }

      for (ParamDeclRefAttr useFromAbove :
           regionDeclUses->second.usesFromAbove) {
        auto decl =
            ParamDeclAttr::get(useFromAbove.getName(), useFromAbove.getType());
        if (capturedParamDecls.insert(decl))
          capturedParamValues.push_back(useFromAbove);
      }

      // Create a wrapper that knows how to handle the global variable. It has
      // the same parameter signature as the lifted region, but it has the same
      // value signature as the original parameter region (no captures - those
      // come from global variables).
      SmallVector<ParamDeclAttr> inputParamDecls(
          capturedParamDecls.getArrayRef());
      llvm::append_range(inputParamDecls, regionDecl.getInputParams());

      SignatureType wrapperSignature = prependParams(
          regionDecl.getSignature(), capturedParamDecls.getArrayRef());

      b.setInsertionPoint(generator);
      auto uniqueName = b.getStringAttr(getUniqueSymbolName(
          (generator.getName() + "_" + regionName).str(), symtab, counter));
      auto liftedWrapper = b.create<GeneratorOp>(
          uniqueName, wrapperSignature, regionDecl.getFunctionType(),
          inputParamDecls, regionDecl.getResultParams(), std::nullopt,
          regionDecl.getInlineLevel(), ExportKind::NotExported,
          b.getDictionaryAttr({}));
      symtab.insert(liftedWrapper);
      auto wrapperSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(liftedWrapper.getNameAttr()), wrapperSignature);

      // Take the body from the param region.
      Region &body = liftedWrapper.getBodyRegion();
      body.takeBody(region);

      // Add the original arguments to the call after the captures. Since the
      // captures are the last N arguments, we can simply drop them.
      b.setInsertionPointToStart(liftedWrapper.getBody());

      // Fill the body of the wrapper.
      for (auto [idx, capture] : llvm::enumerate(captures)) {
        auto load = b.create<POP::CompilerGlobalLoadOp>(
            capture.getType(),
            b.getStringAttr(generator.getName() + "_context_var_" +
                            Twine(varCounter + idx)));
        // HACK: Because we don't track lifetimes of captured variables in
        // parameter closures correctly, we might get erroneous lifetime markers
        // of captured stack allocations. Just clear them out for now.
        for (OpOperand &use : llvm::make_early_inc_range(capture.getUses())) {
          Operation *user = use.getOwner();
          if (body.isAncestor(user->getParentRegion())) {
            if (isa<POP::StackAllocLifetimeStartOp,
                    POP::StackAllocLifetimeEndOp>(user))
              user->eraseOperand(use.getOperandNumber());
            else
              use.set(load);
          }
        }
      }

      // Since the lifted generator will have a new name, we need to update the
      // linkage name in the subprogram information.
      DebugInfo::updateSubprogram(liftedWrapper,
                                  liftedWrapper.getSymNameAttr());

      Attribute bindSignature = wrapperSymbol;
      // If we have parameter captures, create a bind_signature operator.
      if (!capturedParamValues.empty()) {
        // OK cool, now we need a partial binding. First we insert the lifted
        // symbol at the beginning of the vector.
        SmallVector<TypedAttr> partialBindings = {wrapperSymbol};
        llvm::append_range(partialBindings, capturedParamValues);

        // Ignore implicit lifetimes.
        for (Type paramType : regionDecl.getSignature().getInputParamTypes())
          partialBindings.push_back(UnboundAttr::get(paramType));
        bindSignature =
            ParamOperatorAttr::get(POC::BindSignature, partialBindings);
      }

      // Now replace the region decl with a partial binding to the lifted
      // wrapper. The location of the region decl op contains the subprogram
      // scope that itself creates, which we need to override with the
      // parent's scope.
      if (DebugInfo::DIScopeAttr scope =
              DebugInfo::extractScope(regionDecl->getParentOp())) {
        MLIRContext *ctx = scope.getContext();
        Location regionLoc = regionDecl->getLoc();
        // The region may not include a scope if it's always_inline_no_debug.
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(regionLoc))
          b.setLoc(FusedLoc::get(ctx, fusedLoc.getLocations(), scope));
        else
          b.setLoc(FusedLoc::get(ctx, regionLoc, scope));
      }

      // Set the insertion point to the regionDecl.
      b.setInsertionPoint(regionDecl);

      // Create a container for the struct with all the various captures.
      for (auto [idx, capture] : llvm::enumerate(captures)) {
        b.create<POP::CompilerGlobalStoreOp>(
            b.getStringAttr(generator.getName() + "_context_var_" +
                            Twine(varCounter + idx)),
            capture);
      }

      // Create the decl that replaces the regionDecl with its parameter being
      // this new partial binding.
      b.create<ParamDeclareOp>(regionDecl.getParamDecl(),
                               cast<TypedAttr>(bindSignature));

      // And we can drop the regionDecl now, we're done with it.
      toErase.push_back(regionDecl);
      varCounter += captures.size();
    });
    if (hadError)
      return signalPassFailure();

    for (Operation *op : toErase)
      op->erase();
  }
}
