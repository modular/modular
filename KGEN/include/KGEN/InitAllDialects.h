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

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace M {
namespace KGEN {
/// This dialect extension ensures dialects that plug into KGEN's pretty
/// parameter printing are loaded.
class ParameterPrettyFormatExtension : public mlir::DialectExtensionBase {
public:
  /// Apply this extension once the KGEN dialect is loaded.
  explicit ParameterPrettyFormatExtension()
      : DialectExtensionBase(KGENDialect::getDialectNamespace()) {}

  /// Apply the extension by loading all other dialects with pretty printing.
  void apply(MLIRContext *ctx,
             MutableArrayRef<Dialect *> dialects) const override {
    ctx->loadDialect<POP::POPDialect>();
  }

  /// Clone the extension.
  std::unique_ptr<mlir::DialectExtensionBase> clone() const override {
    return std::make_unique<ParameterPrettyFormatExtension>();
  }
};
} // namespace KGEN

// Add all the MLIR dialects to the provided registry.
inline void registerAllKGENDialects(DialectRegistry &registry) {
  registry.insert<HLCF::HLCFDialect>();
  registry.insert<KGEN::KGENDialect>();
  registry.insert<KGEN::LIT::LITDialect>();
  registry.insert<KGEN::POP::POPDialect>();
  registry.insert<MDialect>();
  registry.addExtensions<KGEN::ParameterPrettyFormatExtension>();
}
} // namespace M

#endif // KGEN_INITALLDIALECTS_H
