//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilationOptions.h"
#include "KGEN/ParseLit.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace M;

int main(int argc, char *argv[]) {
  // Register the lit parser.
  llvm::cl::opt<KGEN::CompilationOptions::DebugInfoLevel> debugInfoLevel{
      "import-lit-debug-level",
      llvm::cl::desc("The level of debug info to use during import"),
      llvm::cl::values(
          clEnumValN(KGEN::CompilationOptions::kNoDebug, "none",
                     "Disable all debug info."),
          clEnumValN(KGEN::CompilationOptions::kLineTablesOnly, "line-tables",
                     "Only generate debug info for line number tables."),
          clEnumValN(KGEN::CompilationOptions::kFullDebugInfo, "full",
                     "Generate full debug info.")),
      llvm::cl::init(KGEN::CompilationOptions::kNoDebug)};
  mlir::TranslateToMLIRRegistration fromLit(
      "import-lit", "Import 'lit' from source",
      [&](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        KGEN::CompilationOptions options;
        options.debugLevel = debugInfoLevel;
        return importLitFile(sourceMgr, context, ts, options);
      });

  // Register LLVM IR generation.
  mlir::TranslateFromMLIRRegistration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(os, nullptr);
        return success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<MDialect>();
        mlir::registerLLVMDialectTranslation(registry);
      });

  // Run the tool driver.
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
