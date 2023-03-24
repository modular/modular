//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CLOptions.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/ParseLit.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;

int main(int argc, char *argv[]) {
  KGEN::KGENCommonOptions clOptions;

  cl::opt<bool> validateDocStrings{
      "mojo-validate-doc-strings",
      cl::desc("Validate doc strings in the input .mojo file."),
      cl::init(false)};

  mlir::TranslateToMLIRRegistration fromMojo(
      "import-mojo", "Import 'mojo' from source",
      [&](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        sourceMgr.setIncludeDirs(clOptions.searchPaths);

        // Set up the runtime.
        std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();
        mlir::TimingScope ts;
        KGEN::CompilationOptions options = clOptions.getCompilationOptions();
        return importLitFile(sourceMgr, context, ts, options,
                             /*useMLIRDiagnostics=*/true, *runtime,
                             validateDocStrings);
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
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
      });

  // Run the tool driver.
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
