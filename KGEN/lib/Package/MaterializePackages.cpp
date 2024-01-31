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
#include "Support/Compiler/BytecodeReaderWriter.h"
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
// Registration
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_MATERIALIZEPACKAGES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

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
      if (failed(loadSymbolsFromBytecode(op, reader, symtab, bytecodeSymtab)))
        return signalPassFailure();

    // Finalize the bytecode reader, dropping anything that wasn't
    // materialized.
    if (failed(reader.finalize()))
      return signalPassFailure();
  }
}
