//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INITALLDIALECTS_H
#define KGEN_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace M {
/// Add all the KGEN dialects and extensions to the provided registry.
void registerAllKGENDialects(mlir::DialectRegistry &registry);
} // namespace M

#endif // KGEN_INITALLDIALECTS_H
