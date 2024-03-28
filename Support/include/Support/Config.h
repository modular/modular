//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONFIG_H
#define SUPPORT_CONFIG_H

namespace mlir {
class MLIRContext;
class PassManager;
} // namespace mlir

namespace M {
/// This function configures the MLIR context according to the current build
/// configuration. In modular production builds, it disables IR dumps in
/// diagnostics.
void configureMLIRContext(mlir::MLIRContext &ctx);

/// This function configures the MLIR pass manager according to the current
/// build configuration. In modular production builds, it disables verification
/// after all passes.
void configurePassManager(mlir::PassManager &mgr);
} // namespace M

#endif // SUPPORT_CONFIG_H
