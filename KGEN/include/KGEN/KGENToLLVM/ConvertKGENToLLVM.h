//===- KGEN/KGENToLLVM/ConvertKGENToLLVM.h --------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H
#define KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace M::KGEN {
class KGENToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  KGENToLLVMTypeConverter(mlir::Location loc);

  /// Report an error or conversion failure.
  /// TODO: TypeConverter needs an error reporting mechanism.
  mlir::InFlightDiagnostic emitError(llvm::StringRef msg) {
    return mlir::emitError(loc) << msg;
  }

private:
  /// A location used to report conversion failures.
  mlir::Location loc;
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateKGENToLLVMPatterns(KGENToLLVMTypeConverter &typeConverter,
                                mlir::RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createConvertKGENToLLVMPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENToLLVM/ConvertKGENToLLVM.h.inc"
} // namespace M::KGEN

#endif // KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H
