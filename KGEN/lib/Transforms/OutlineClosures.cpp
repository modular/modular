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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
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
  SymbolTable symtab(theModule);

  // Walk over all the param.declare.region ops and create structs with the SSA
  // captures, use bind_signature to deal with parameter captures.
  OpBuilder b(theModule);
  b.setInsertionPointToStart(theModule.getBody());

  unsigned counter = 0, varCounter = 0;
  for (auto generator : theModule.getOps<GeneratorOp>()) {
    // Calculate the parameter decls and uses for the region decl's parent.
    ParameterDeclsAndUses uses;
    auto subregions = uses.calculate(generator);

    // We'll use this a lot here - pull it out into a little lambda.
    auto getUniqueName = [&](StringRef suffix) {
      return b.getStringAttr(getUniqueSymbolName(
          (generator.getName() + suffix).str(), symtab, counter));
    };

    auto getUniqueVarName = [&](StringRef suffix) {
      return b.getStringAttr(generator.getName() + suffix + "_" +
                             Twine(varCounter++));
    };

    for (auto regionDecl :
         llvm::make_early_inc_range(generator.getOps<ParamDeclareRegionOp>())) {
      LLVM_DEBUG(llvm::dbgs() << "Lifting closure: " << regionDecl << "\n");
      // Value captures are easy (ish)
      SmallVector<Value> captures;
      bool isolated = M::operationIsIsolatedFromAbove(regionDecl, &captures);
      // Check the captures for any types that refer to parameters. For right
      // now, these are the *only* parameters we care about - we're filling out
      // the parameter captures for the struct.
      llvm::SetVector<ParamDeclAttr> necessaryDecls;
      for (Value capture : captures) {
        // Check if the type of the capture is parametrized. If it is, add a
        // decl to the list of necessary decls.
        if (auto subElts =
                dyn_cast<mlir::SubElementTypeInterface>(capture.getType()))
          subElts.walkSubAttrs([&](Attribute attr) {
            if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr))
              necessaryDecls.insert(
                  ParamDeclAttr::get(declRef.getName(), declRef.getType()));
          });
      }

      LLVM_DEBUG(llvm::dbgs() << "Found value captures: [";
                 llvm::interleaveComma(captures, llvm::dbgs());
                 llvm::dbgs() << "]\n");

      // Create a struct with the correct parameter decls.
      b.setInsertionPoint(generator);
      auto structDecl = b.create<StructDeclOp>(
          regionDecl.getLoc(), getUniqueName("_context"),
          ParamDeclArrayAttr::get(regionDecl.getContext(),
                                  necessaryDecls.getArrayRef()));
      symtab.insert(structDecl);

      // Create a field for each capture.
      structDecl.getFields().push_back(new Block);
      b.setInsertionPointToStart(&structDecl.getFields().front());
      for (auto [idx, capture] : llvm::enumerate(captures)) {
        b.create<StructFieldOp>(capture.getLoc(),
                                b.getStringAttr("field_" + Twine(idx)),
                                TypeAttr::get(capture.getType()));
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Created capture struct: " << structDecl << "\n");

      // 'Create' a global variable with an instance of the struct decl (really
      // just a StringAttr). We need to get the parameter bindings for each
      // parameter the struct declares.
      auto globalVar = getUniqueVarName("_context_var");
      SmallVector<ParamBindAttr> structTypeBindings;
      for (ParamDeclAttr decl : structDecl.getInputParamDecls())
        structTypeBindings.push_back(ParamBindAttr::get(
            decl, ParamDeclRefAttr::get(decl.getName(), decl.getType())));

      auto globalVarType =
          DeclRefType::get(SymbolRefAttr::get(structDecl.getNameAttr()),
                           b.getAttr<ParamBindArrayAttr>(structTypeBindings));

      auto body = cast<RegionBodyOp>(regionDecl.getBody().front().front());

      // Collect any parameters used from above that we need to capture for the
      // lifted generator.
      auto regionDeclUses = subregions.find(body);
      SmallVector<TypedAttr> partialBindings;
      if (regionDeclUses != subregions.end()) {
        for (ParamDeclRefAttr useFromAbove :
             regionDeclUses->getSecond().usesFromAbove) {
          necessaryDecls.insert(ParamDeclAttr::get(useFromAbove.getName(),
                                                   useFromAbove.getType()));
          // Create a binding that just references the attr we already have.
          partialBindings.push_back(useFromAbove);
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "Found parameter captures: [";
                 llvm::interleaveComma(partialBindings, llvm::dbgs());
                 llvm::dbgs() << "]\n");

      SignatureType bodySignature = body.getFullSignature();

      // The value signature is pretty simple here, just captures and then any
      // original arguments.
      SmallVector<Value> liftedInputs = captures;
      llvm::append_range(liftedInputs, body.getBodyRegion().getArguments());
      auto liftedValueSignature =
          FunctionType::get(&getContext(), ValueRange(liftedInputs).getTypes(),
                            bodySignature.getValueResults());

      // The parameter signature is just the necessary decls + original
      // arguments, and then any of the original results.
      for (ParamDeclAttr inputParam : body.getInputParamDecls())
        necessaryDecls.insert(inputParam);

      // Pull together the input conventions - all the captures all use the
      // default convention (despite what the enum says).
      SmallVector<ValueInputConvention> liftedConventions(
          captures.size(), ValueInputConvention::ByVal);
      llvm::append_range(liftedConventions,
                         bodySignature.getValueInputConventions());

      // The lifted generator needs to be force_inline, so we add that to the
      // FnEffects.
      auto liftedSignature = SignatureType::get(
          b.getAttr<ParamDeclArrayAttr>(necessaryDecls.getArrayRef()),
          bodySignature.getResultParamTypes(), liftedValueSignature,
          b.getAttr<ConventionsAttr>(liftedConventions,
                                     bodySignature.getFnEffects()));

      // Now lift the body out into its own generator.
      b.setInsertionPoint(generator);
      auto lifted = b.create<GeneratorOp>(
          regionDecl.getLoc(), getUniqueName(""),
          TypeAttr::get(liftedSignature),
          b.getAttr<ConstraintArrayAttr>(ArrayRef<ConstraintAttr>{}),
          FlatSymbolRefAttr());
      symtab.insert(lifted);
      auto liftedSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(lifted.getSymNameAttr()), liftedSignature);

      // Create the generator's body.
      if (!isolated) {
        // Not isolated, so we have to clone the ops in so we can remap
        // arguments.
        auto *newBody = new Block;
        BlockAndValueMapping map;
        for (Value capture : captures)
          map.map(capture,
                  newBody->addArgument(capture.getType(), capture.getLoc()));

        b.setInsertionPointToStart(newBody);
        for (Operation &op : *body.getBody())
          b.clone(op, map);

        lifted.getBodyRegion().push_back(newBody);
      } else {
        // Take the body from the param region.
        lifted.getBodyRegion().takeBody(body.getBodyRegion());
      }
      LLVM_DEBUG(llvm::dbgs() << "Created lifted region: " << lifted << "\n");

      // Create a wrapper that knows how to handle the global variable. It has
      // the same parameter signature as the lifted region, but it has the same
      // value signature as the original parameter region (no captures - those
      // come from global variables).
      auto wrapperSignature = SignatureType::get(
          liftedSignature.getInputParams(),
          liftedSignature.getResultParamTypes(), bodySignature.getValues(),
          bodySignature.getConventions());

      b.setInsertionPoint(generator);
      auto liftedWrapper = b.create<GeneratorOp>(
          regionDecl.getLoc(), getUniqueName("_wrapper"),
          TypeAttr::get(wrapperSignature),
          b.getAttr<ConstraintArrayAttr>(ArrayRef<ConstraintAttr>{}),
          FlatSymbolRefAttr());
      symtab.insert(liftedWrapper);
      auto wrapperSymbol = SymbolConstantAttr::get(
          SymbolRefAttr::get(liftedWrapper.getNameAttr()), wrapperSignature);

      // Fill the body of the wrapper.
      liftedWrapper.getBodyRegion().push_back(new Block);
      b.setInsertionPointToStart(liftedWrapper.getBody());
      auto load = b.create<POP::CompilerGlobalLoadOp>(regionDecl.getLoc(),
                                                      globalVarType, globalVar);
      // Create accesses for each capture.
      SmallVector<Value> callArgs;
      for (StructFieldOp structField : structDecl.getFieldDecls()) {
        callArgs.push_back(
            b.create<StructExtractOp>(structField.getLoc(), load, structField));
      }

      // Add the original arguments to the call after the captures.
      for (BlockArgument originalArg : body.getBodyRegion().getArguments()) {
        callArgs.push_back(liftedWrapper.getBodyRegion().addArgument(
            originalArg.getType(), originalArg.getLoc()));
      }

      // Create result parameter decls from the lifted region, and get decl refs
      // for the actual ReturnOp.
      SmallVector<ParamDeclAttr> resultDecls;
      SmallVector<TypedAttr> returnRefs;
      for (auto [idx, resultParamTy] :
           llvm::enumerate(lifted.getResultParamTypes())) {
        auto declName = b.getStringAttr("__resultParam_" + Twine(idx));
        // If something is somehow named __resultParam_0 then just increment the
        // counter till it works.
        while (
            llvm::find_if(lifted.getInputParamDecls(), [&](ParamDeclAttr decl) {
              return decl.getName() == declName;
            }) != lifted.getInputParamDecls().end()) {
          declName = b.getStringAttr("__resultParam_" + Twine(++idx));
        }

        resultDecls.push_back(ParamDeclAttr::get(declName, resultParamTy));
        returnRefs.push_back(ParamDeclRefAttr::get(declName, resultParamTy));
      }

      // We need to set the parameter bindings for the call to the lifted
      // region. This basically just means binding the wrapper's input params to
      // a ref.
      SmallVector<ParamBindAttr> symbolBindings;
      for (ParamDeclAttr decl : liftedWrapper.getInputParamDecls()) {
        symbolBindings.push_back(ParamBindAttr::get(
            decl, ParamDeclRefAttr::get(decl.getName(), decl.getType())));
      }

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
      b.create<ReturnOp>(regionDecl.getLoc(), returnRefs,
                         ValueRange(callResults.getResults()));

      // OK cool, now we need a partial binding. First we insert the lifted
      // symbol at the beginning of the vector.
      partialBindings.insert(partialBindings.begin(), wrapperSymbol);
      // Then we add `#kgen.unbound` for the things we aren't binding.
      for (size_t i = partialBindings.size() - 1,
                  e = liftedSignature.getInputParams().size();
           i < e; ++i) {
        partialBindings.push_back(
            UnboundAttr::get(liftedSignature.getInputParams()[i].getType()));
      }

      Attribute bindSignature =
          ParamOperatorAttr::get(POC::BindSignature, partialBindings);

      // Now replace the region decl with a partial binding to the lifted
      // wrapper.
      b.setInsertionPoint(regionDecl);
      // Create a container for the struct with all the various captures.
      auto container = b.create<StructCreateOp>(regionDecl.getLoc(),
                                                globalVarType, captures);

      // Get a pointer to the global and store the container in it.
      b.create<POP::CompilerGlobalStoreOp>(regionDecl.getLoc(), globalVar,
                                           container);

      // Create the decl that replaces the regionDecl with its parameter being
      // this new partial binding.
      b.create<ParamDeclareOp>(regionDecl.getLoc(),
                               regionDecl.getParamDecls().front(),
                               bindSignature);

      // And we can drop the regionDecl now, we're done with it.
      regionDecl->erase();
    }
  }
}
