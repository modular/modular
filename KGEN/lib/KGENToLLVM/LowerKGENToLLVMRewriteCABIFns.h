//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Rewrites abi("C") LLVM function definitions to use the platform C ABI
// (CABIInfo). Separated from LowerKGENToLLVM.cpp so that reviewers can read
// the definition-side logic in isolation from the call-site patterns.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
#define KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H

namespace mlir::LLVM {
class LLVMFuncOp;
} // namespace mlir::LLVM

namespace M::KGEN {
class CABIInfo;

/// Rewrite a abi("C") LLVM function definition to use the platform C ABI.
/// Applies entry-block argument coercion (register/indirect/two-register) and
/// return-value coercion (register/sret) in place, then updates the function
/// type signature. No-ops for external declarations and identity-ABI functions.
/// Mojo direct callers are already patched by ConvertKGENCall.
void processCABIFunctionDefinition(mlir::LLVM::LLVMFuncOp func,
                                   CABIInfo &abiInfo);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERKGENTOLLVMREWRITECABIFNS_H
