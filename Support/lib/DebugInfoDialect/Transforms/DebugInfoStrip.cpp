//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DEF_DEBUGINFOSTRIP
#include "Support/DebugInfoDialect/Transforms/Transforms.h.inc"
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoStrip : public impl::DebugInfoStripBase<DebugInfoStrip> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void DebugInfoStrip::runOnOperation() {
  mlir::AttrTypeReplacer replacer;

  // If we're preserving line tables, we need to replace the compile unit
  // attribute with one that only contains line tables.
  if (preserveLineTables) {
    replacer.addReplacement(
        [](DebugInfo::DICompileUnitAttr CU) -> std::optional<Attribute> {
          if (CU.getEmissionKind() == DebugInfo::EmissionKind::Full) {
            return DebugInfo::DICompileUnitAttr::get(
                CU.getSourceLanguage(), CU.getFile(), CU.getProducer(),
                CU.getIsOptimized(), DebugInfo::EmissionKind::LineTablesOnly,
                CU.getNameTableKind());
          }
          return std::nullopt;
        });

    // Otherwise, we strip debug info from locations.
  } else {
    replacer.addReplacement(
        [&](mlir::FusedLocWith<DIAttr> diLoc) -> mlir::LocationAttr {
          return FusedLoc::get(diLoc.getContext(), diLoc.getLocations());
        });
  }

  getOperation()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Drop all debug info operations.
    if (isa_and_nonnull<DebugInfoDialect>(op->getDialect())) {
      op->erase();
      return WalkResult::skip();
    }

    // For everything else, update the location.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/false,
                               /*replaceLocs=*/true);
    return WalkResult::advance();
  });
}
