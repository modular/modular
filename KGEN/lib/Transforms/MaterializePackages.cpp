//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_MATERIALIZEPACKAGES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Inflation
//===----------------------------------------------------------------------===//

/// Read the given operation from bytecode, and pull in any of its dependencies.
static LogicalResult readFromBytecode(Operation *op,
                                      mlir::BytecodeReader &reader,
                                      SymbolTable &symTab,
                                      const SymbolTable &bytecodeSymTab) {
  if (reader.isMaterializable(op)) {
    if (failed(reader.materialize(op, [&](Operation *op) { return true; })))
      return failure();
  }

  // Extract a dependency from the bytecode module and move it into the main
  // module, if it doesn't already exist there. If a symbol was moved, return
  // it.
  auto extractDependency = [&](StringAttr name) -> Operation * {
    // Don't move the symbol if it already exists in the main module.
    if (symTab.lookup(name))
      return nullptr;
    Operation *symbol = bytecodeSymTab.lookup(name);
    assert(symbol && "expected valid symbol reference");

    // Move the symbol into the main module.
    symbol->moveAfter(op);
    symTab.insert(symbol);
    return symbol;
  };

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) -> WalkResult {
    if (Operation *decl = extractDependency(ref.getAttr()))
      return readFromBytecode(decl, reader, symTab, bytecodeSymTab);
    return WalkResult::advance();
  });
  auto result = op->walk([&](Operation *op) {
    // Extract references to type declarations.
    if (walker.walk(op->getAttrDictionary()).wasInterrupted())
      return WalkResult::interrupt();
    for (Type type : op->getResultTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        if (walker.walk(type).wasInterrupted())
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct MaterializePackagesPass
    : impl::MaterializePackagesBase<MaterializePackagesPass> {
  void runOnOperation() override;
};
} // namespace

void MaterializePackagesPass::runOnOperation() {
  auto theModule = cast<ModuleOp>(getOperation());
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Collect the functions that need inflation, and the source containing
  // their bodies.
  llvm::MapVector<StringAttr, SmallVector<ExternGeneratorOp>> toInflate;
  for (auto func : theModule.getOps<ExternGeneratorOp>())
    toInflate[func.getPreCompiledModuleRefAttr().getAttr()].emplace_back(func);

  mlir::ParserConfig parserConfig(&getContext(), /*verifyAfterParse=*/false);
  for (auto &[moduleRef, funcs] : toInflate) {
    auto packageLink = symtab.lookup<PackageLinkOp>(moduleRef);
    if (!packageLink) {
      funcs[0].emitOpError("unable to find the link for preCompiledModuleRef");
      return signalPassFailure();
    }
    TargetInfoAttr compiledFor = packageLink.getArchive().getTarget();
    DenseResourceElementsAttr precompiledBody =
        packageLink.getArchive().getElaboratedModule();
    bool isPreElaborated = true;

    // Get the target on the module. If we don't have a target or if the
    // target doesn't match, check for a fallback where we can compile the
    // package right now on-demand.
    TargetInfoAttr target = M::lookupTargetInfo(theModule);
    if (!target || target != compiledFor) {
      // If we have the pre-elaboration module, just recompile the package.
      // Otherwise, emit an error that we can't use this package.
      precompiledBody = packageLink.getPreElaborationModuleAttr();
      if (!precompiledBody) {
        auto diag = packageLink.emitError("package was compiled for ")
                    << compiledFor << " but current target is " << target;
        diag.attachNote() << "no generic fallback was found to recompile "
                             "package for current target";
        return signalPassFailure();
      }
      isPreElaborated = false;
    }

    // Get the data for the precompiled body.
    mlir::AsmResourceBlob *blob = precompiledBody.getRawHandle().getBlob();
    if (!blob) {
      funcs[0].emitError("unable to find the precompiled body blob");
      return signalPassFailure();
    }
    ArrayRef<char> bytecode = blob->getData();
    llvm::MemoryBufferRef bufferRef(
        StringRef(bytecode.begin(), bytecode.size()), "");

    // Start lazy loading the bytecode for the function bodies.
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    mlir::BytecodeReader reader(bufferRef, parserConfig, /*lazyLoad=*/true,
                                sourceMgr);
    Block block;
    if (failed(reader.readTopLevel(&block)))
      return signalPassFailure();
    ModuleOp bytecodeModule = cast<ModuleOp>(block.front());
    if (failed(reader.materialize(bytecodeModule)))
      return signalPassFailure();

    // If we're "inflating" pre-elaborated ops into the current module, then
    // make sure they're not exported -- the user exports their own functions.
    if (!isPreElaborated) {
      SmallVector<std::pair<ExportInterface, ExportKind>> exportKinds;
      for (ExportInterface op : bytecodeModule.getOps<ExportInterface>()) {
        auto exportKind = op.getExportKind();
        if (exportKind != ExportKind::NotExported) {
          exportKinds.push_back({op, exportKind});
          op.setNotExported();
        }
      }
    }

    // Collect the symbols within the bytecode.
    SymbolTable bytecodeSymtab(cast<ModuleOp>(block.front()));

    // Replace the high level functions with the precompiled counter parts in
    // the bytecode module.
    SmallVector<Operation *> operationsToInflate;
    for (ExternGeneratorOp func : funcs) {
      auto result = bytecodeSymtab.lookup<ExportInterface>(func.getName());
      if (!result) {
        func.emitError() << "unable to find " << func.getName()
                         << " in precompiled bytecode";
        return signalPassFailure();
      }
      operationsToInflate.push_back(result);
      result->moveAfter(func);

      // Replace the original function with the parsed KGEN Func.
      symtab.erase(func);
      symtab.insert(result);
    }

    // Now that we've replaced the high level functions with the bytecode
    // functions, inflate them and pull in all of the dependencies.
    for (Operation *op : operationsToInflate)
      if (failed(readFromBytecode(op, reader, symtab, bytecodeSymtab)))
        return signalPassFailure();

    // Finalize the bytecode reader, dropping anything that wasn't
    // materialized.
    if (failed(reader.finalize()))
      return signalPassFailure();

    // Convert the package link to a kgen link directive if we're using the
    // fully compiled package. If we're recompiling the package, just drop the
    // package link altogether.
    if (isPreElaborated) {
      OpBuilder b(packageLink);
      auto linkOp = b.create<KGEN::LinkOp>(
          packageLink.getLoc(), packageLink.getSymNameAttr(), StringAttr(),
          packageLink.getArchive().getArchive());
      symtab.erase(packageLink);
      symtab.insert(linkOp);
    } else {
      symtab.erase(packageLink);
    }
  }
}
