//===- KGEN/KGENToLLVM/ConvertKGENToLLVM.h --------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H
#define KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace M {
class KGENToLLVMTypeConverter : public mlir::TypeConverter {
public:
  KGENToLLVMTypeConverter();
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
} // namespace M

#endif // KGEN_KGENTOLLVM_CONVERTKGENTOLLVM_H
