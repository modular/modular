//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/SnapshotDebugInfo.h"

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
  replacer.addReplacement(
      [&](mlir::FusedLocWith<DIAttr> diLoc) -> mlir::LocationAttr {
        return FusedLoc::get(diLoc.getContext(), diLoc.getLocations());
      });

  getOperation()->walk([&](Operation *op) {
    // Drop all debug info operations.
    if (isa_and_nonnull<DebugInfoDialect>(op->getDialect()))
      return op->erase();

    // For everything else, update the location.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/false,
                               /*replaceLocs=*/true);
  });
}
