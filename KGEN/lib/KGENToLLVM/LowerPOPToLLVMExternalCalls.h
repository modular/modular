//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H
#define KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H

namespace mlir {
class RewritePatternSet;
class SymbolTable;
} // namespace mlir

namespace M::KGEN {

struct POPToLLVMTypeConverter;

/// Register the pattern that lowers POP external_call ops to LLVM calls,
/// applying platform-specific C ABI struct coercion when necessary.
///
/// The pattern handles struct argument/return coercion for x86-64 System V
/// and ARM64 AAPCS, falling back to pass-through for other targets.
void populateLowerPOPExternalCallPatterns(mlir::RewritePatternSet &patterns,
                                          POPToLLVMTypeConverter &typeConverter,
                                          mlir::SymbolTable &symtab);

} // namespace M::KGEN

#endif // KGEN_LIB_KGENTOLLVM_LOWERPOPTOLLVMEXTERNALCALLS_H
