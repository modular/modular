//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ParseLit.h"
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
  mlir::TranslateToMLIRRegistration fromLit(
      "import-lit", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importLitFile(sourceMgr, context, ts);
      });

  // Register LLVM IR generation.
  mlir::TranslateFromMLIRRegistration(
      "mlir-to-llvmir",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(os, nullptr);
        return success();
      },
      [](mlir::DialectRegistry &registry) {
        mlir::registerLLVMDialectTranslation(registry);
      });

  // Run the tool driver.
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
