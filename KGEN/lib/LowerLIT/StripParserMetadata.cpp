//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/LITDialect/LITOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_STRIPPARSERMETADATA
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct StripParserMetadataPass
    : public impl::StripParserMetadataBase<StripParserMetadataPass> {
  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      // Strip doc strings from ASTDecl operations.
      if (auto astDecl = dyn_cast<LIT::ASTDeclInterface>(op))
        astDecl.removeDocStringAttr();
    });
  }
};
} // namespace
