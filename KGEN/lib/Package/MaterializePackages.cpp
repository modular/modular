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
#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// loadPreElaboratedBytecodeForLinking
//===----------------------------------------------------------------------===//

// FIXME: Move the declarations of these functions into a separate util header.
ErrorOr<OwningOpRef<ModuleOp>> M::KGEN::loadPreElaboratedModuleBytecode(
    DenseResourceElementsAttr bytecodeAttr) {
  mlir::AsmResourceBlob *blob = bytecodeAttr.getRawHandle().getBlob();
  if (!blob)
    return Error("unable to find the pre-elaborated module blob");
  ArrayRef<char> bytecode = blob->getData();
  llvm::MemoryBufferRef bufferRef(StringRef(bytecode.begin(), bytecode.size()),
                                  "");

  // Read the entirety of the bytecode in the buffer; no lazy loading.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::ParserConfig parserConfig(bytecodeAttr.getContext());
  mlir::BytecodeReader reader(bufferRef, parserConfig, /*lazyLoad=*/false,
                              sourceMgr);
  Block block;
  if (failed(reader.readTopLevel(&block)))
    return Error("unable to read pre-elaborated module blob");

  // Take ownership of the parsed module by removing it from the block so that
  // we can return it.
  ModuleOp module = cast<ModuleOp>(block.front());
  module->remove();
  return module;
}

ErrorOr<OwningOpRef<ModuleOp>> M::KGEN::loadPreElaboratedModuleForLinking(
    DenseResourceElementsAttr bytecodeAttr) {
  ErrorOr<OwningOpRef<ModuleOp>> packageModuleOr =
      loadPreElaboratedModuleBytecode(bytecodeAttr);
  if (packageModuleOr.isError())
    return packageModuleOr.takeError();

  // Strip the implicit package exports, we don't need these because we're going
  // to link the package into an existing module as-is.
  for (ExportInterface op : (*packageModuleOr)->getOps<ExportInterface>())
    if (op.isPackageExported())
      op.setNotExported();

  // Materialize all of the dependent packages now, making sure they also
  // get linked in properly.
  mlir::PassManager pm((*packageModuleOr)->getContext());
  pm.addPass(createMaterializePackages());
  if (failed(pm.run(**packageModuleOr)))
    return Error("unable to materialize dependent packages");

  return std::move(*packageModuleOr);
}

ErrorOr<DenseResourceElementsAttr> M::KGEN::loadPreElaboratedBytecodeForLinking(
    DenseResourceElementsAttr bytecodeAttr) {
  ErrorOr<OwningOpRef<ModuleOp>> packageModuleOr =
      loadPreElaboratedModuleForLinking(bytecodeAttr);
  if (packageModuleOr.isError())
    return packageModuleOr.takeError();

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(**packageModuleOr, *str)))
    return Error("could not write bytecode for package module");

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  return createResourceAttr((*packageModuleOr)->getContext(), str->getBuffer(),
                            "bytecode_" +
                                llvm::toHex(hash, /*LowerCase=*/true));
}

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
LogicalResult M::KGEN::readFromBytecode(Operation *op,
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
  using MaterializePackagesBase::MaterializePackagesBase;

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

    ErrorOr<DenseResourceElementsAttr> bytecodeOr =
        loadPreElaboratedBytecodeForLinking(
            packageLink.getPreElaborationModuleAttr());
    if (bytecodeOr.isError()) {
      mlir::emitError(packageLink.getLoc(),
                      "failed to load precompiled module and its dependencies "
                      "for this package");
      return signalPassFailure();
    }
    DenseResourceElementsAttr bytecode = bytecodeOr.takeValue();

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
    SymbolTable bytecodeSymtab(bytecodeModule);

    // Replace the high level functions with their counter parts in the package
    // bytecode module.
    SmallVector<Operation *> operationsToInflate;
    for (ExternGeneratorOp func : funcs) {
      // Try the post-elaboration (linkage) name first, and then fallback to the
      // pre-elaboration name.
      StringAttr preElaborationName = func.getSymNameAttr();
      auto result = bytecodeSymtab.lookup<GeneratorOp>(preElaborationName);
      if (!result) {
        (void)reader.finalize();
        mlir::emitError(func.getLoc(), "unable to find ")
            << preElaborationName.getValue() << " in imported package bytecode";
        return signalPassFailure();
      }
      operationsToInflate.push_back(result);
      result->moveAfter(func);

      // Propagate the precompiled reference to the materialized generator to
      // indicate that it has an external implementation.
      result.setPreCompiledModuleRefAttr(func.getPreCompiledModuleRefAttr());

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
  }
}
