//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/Package/Package.h"
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
class MaterializePackagesPass
    : public impl::MaterializePackagesBase<MaterializePackagesPass> {
public:
  explicit MaterializePackagesPass(
      PackageLinkHandlerFn packageLinkHandlerFn = nullptr)
      : MaterializePackagesBase(), packageLinkHandlerFn(packageLinkHandlerFn) {}

  LogicalResult initialize(MLIRContext *context) override {
    if (!packageLinkHandlerFn)
      packageLinkHandlerFn = [](PackageLinkOp packageLink,
                                TargetInfoAttr targetInfo) {
        packageLink.emitError("package link could not be handled for target ")
            << targetInfo.getTripleStr();
        return Error("package link handler is null");
      };

    return success();
  }

  void runOnOperation() override;

private:
  PackageLinkHandlerFn packageLinkHandlerFn;
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
  DenseMap<StringAttr, StringAttr> renamedSymbols;
  for (auto &[moduleRef, funcs] : toInflate) {
    auto packageLink = symtab.lookup<PackageLinkOp>(moduleRef);
    if (!packageLink) {
      funcs[0].emitOpError("unable to find the link for preCompiledModuleRef");
      return signalPassFailure();
    }

    TargetInfoAttr target = lookupTargetInfo(theModule);

    // Get the target on the module. If we don't have a target or if the target
    // doesn't match, check for a pre-elaborated module that we can then compile
    // on-demand.
    std::optional<PackageArchiveAttr> archive =
        packageLink.getArchivesAttr().getTargetArchive(target);

    DenseResourceElementsAttr bytecode;
    if (archive) {
      bytecode = archive->getElaboratedModule();
    } else {
      // If we don't have a precompiled archive for our target, then we can only
      // proceed if we have a pre-elaborated module.
      if (!packageLink.getPreElaborationModuleAttr()) {
        auto diag =
            packageLink.emitError(
                "no precompiled package archives match the current target '")
            << target
            << "', archives we precompiled for the following targets: "
            << packageLink.getArchivesAttr().getTargetsAsString();
        diag.attachNote() << "no generic fallback was found to recompile "
                             "package for current target";
        return signalPassFailure();
      }

      // The callback function is given the pre-elaborated module and returns
      // the package module bytecode that is to be imported into the module
      // currently being built.
      ErrorOr<DenseResourceElementsAttr> bytecodeOr =
          packageLinkHandlerFn(packageLink, target);
      if (bytecodeOr.isError()) {
        packageLink.emitError(bytecodeOr.getError());
        return signalPassFailure();
      }
      bytecode = *bytecodeOr;

      // If the callback function elaborated and compiled the package module
      // bytecode, it'll have set it on the `kgen.package_link` op.
      archive = packageLink.getArchivesAttr().getTargetArchive(target);
    }

    // Get the data for the imported module body.
    mlir::AsmResourceBlob *blob = bytecode.getRawHandle().getBlob();
    if (!blob) {
      funcs[0].emitError("unable to find the precompiled body blob");
      return signalPassFailure();
    }
    ArrayRef<char> bytecodeData = blob->getData();
    llvm::MemoryBufferRef bufferRef(
        StringRef(bytecodeData.begin(), bytecodeData.size()), "");

    // Start lazy loading the bytecode for the function bodies.
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    mlir::BytecodeReader reader(bufferRef, parserConfig, /*lazyLoad=*/true,
                                sourceMgr);
    Block block;
    if (failed(reader.readTopLevel(&block))) {
      (void)reader.finalize();
      return signalPassFailure();
    }
    ModuleOp bytecodeModule = cast<ModuleOp>(block.front());
    if (failed(reader.materialize(bytecodeModule))) {
      (void)reader.finalize();
      return signalPassFailure();
    }

    // Collect the symbols within the bytecode.
    SymbolTable bytecodeSymtab(cast<ModuleOp>(block.front()));

    // Replace the high level functions with their counter parts in the package
    // bytecode module.
    SmallVector<Operation *> operationsToInflate;
    for (ExternGeneratorOp func : funcs) {
      // Try the post-elaboration (linkage) name first, and then fallback to the
      // pre-elaboration name.
      StringAttr linkageName = func.getLinkageNameAttr();
      StringAttr preElaborationName = func.getSymNameAttr();
      auto result = bytecodeSymtab.lookup<ExportInterface>(linkageName);
      if (!result) {
        result = bytecodeSymtab.lookup<ExportInterface>(preElaborationName);
        if (!result) {
          (void)reader.finalize();
          mlir::emitError(func.getLoc(), "unable to find ")
              << preElaborationName.getValue()
              << " in imported package bytecode";
          return signalPassFailure();
        }
      } else {
        // The post-elaboration name is different, so we need to do a rename.
        renamedSymbols.insert({func.getSymNameAttr(), linkageName});
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

    // If we read in the elaborated functions, convert the `kgen.package_link`
    // op to a `kgen.link` op`, signifying that the package is now linked in as
    // a library. Otherwise, erase the `kgen.package_link` op altogether -- the
    // imported functions now exist as part of the module into which they were
    // imported.
    if (archive) {
      // Convert the package link to a kgen link directive, passing along the
      // transitive dependencies.
      OpBuilder b(packageLink);
      auto linkOp = b.create<KGEN::LinkOp>(
          packageLink.getLoc(), packageLink.getSymNameAttr(),
          archive->getArchive(), archive->getDependencies());
      symtab.erase(packageLink);
      symtab.insert(linkOp);
    } else {
      symtab.erase(packageLink);
    }
  }
  // Rename post-elaboration references.
  mlir::parallelForEach(
      &getContext(), getOperation().getOps(), [&renamedSymbols](Operation &op) {
        mlir::AttrTypeReplacer replacer;
        replacer.addReplacement([&renamedSymbols](FlatSymbolRefAttr ref) {
          if (auto it = renamedSymbols.find(ref.getAttr());
              it != renamedSymbols.end())
            return FlatSymbolRefAttr::get(it->second);
          return ref;
        });
        replacer.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/true,
                                              /*replaceLocs=*/false,
                                              /*replaceTypes=*/true);
      });
}

std::unique_ptr<mlir::Pass>
M::KGEN::createMaterializePackages(PackageLinkHandlerFn packageLinkHandlerFn) {
  return std::make_unique<MaterializePackagesPass>(packageLinkHandlerFn);
}
