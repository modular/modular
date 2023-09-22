//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_INITALLDIALECTS_H
#define KGEN_TOOLCOMMON_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace M {
/// Add all the KGEN dialects and extensions to the provided registry.
void registerAllKGENDialects(mlir::DialectRegistry &registry);
/// Register all required LLVMIR translation interfaces.
void registerKGENToLLVMTranslation(mlir::DialectRegistry &registry);
} // namespace M

#endif // KGEN_TOOLCOMMON_INITALLDIALECTS_H
