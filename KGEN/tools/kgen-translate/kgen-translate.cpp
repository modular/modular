//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

int main(int argc, char *argv[]) {
  mlir::TranslateFromMLIRRegistration(
      "mlir-to-llvmir",
      [](mlir::ModuleOp module, llvm::raw_ostream &os) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
          return mlir::failure();

        llvmModule->print(os, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        mlir::registerLLVMDialectTranslation(registry);
      });
  return failed(mlir::mlirTranslateMain(argc, argv, "KGEN Translate Tool"));
}
