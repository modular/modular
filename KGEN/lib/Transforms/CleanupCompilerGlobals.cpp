//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_CLEANUPCOMPILERGLOBALS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct CleanupCompilerGlobalsPass
    : M::KGEN::impl::CleanupCompilerGlobalsBase<CleanupCompilerGlobalsPass> {
  void runOnOperation() override;
};
} // namespace

void CleanupCompilerGlobalsPass::runOnOperation() {
  FuncOp func = getOperation();

  // When we see a sequence of store->load we can just remove it and replace it
  // with the store argument. Process the loads and stores according to program
  // order.
  DenseMap<StringAttr, std::pair<mlir::LocationAttr, Value>> values;
  WalkResult result = func.walk([&](Operation *op) -> WalkResult {
    if (auto load = dyn_cast<POP::CompilerGlobalLoadOp>(op)) {
      auto it = values.find(load.getNameAttr());
      if (it == values.end())
        return WalkResult::advance();
      auto [loc, value] = it->second;
      if (value.getType() != load.getType()) {
        (load.emitError("compiler global load type does not match "
                        "previous store type"))
                .attachNote(loc)
            << "see previous store to variable here";
        return WalkResult::interrupt();
      }
      load.replaceAllUsesWith(it->second.second);
      load.erase();
    } else if (auto store = dyn_cast<POP::CompilerGlobalStoreOp>(op)) {
      values[store.getNameAttr()] =
          std::make_pair(store.getLoc(), store.getValue());
      store.erase();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
