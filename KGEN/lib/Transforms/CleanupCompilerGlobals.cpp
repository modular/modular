//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"

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
  // with the store argument.
  DenseMap<SymbolRefAttr, std::pair<POP::CompilerGlobalStoreOp, bool>> stores;
  for (Operation &op : llvm::make_early_inc_range(func.getOps())) {
    if (auto store = dyn_cast<POP::CompilerGlobalStoreOp>(op)) {
      stores[store.getName()] = {store, false};
      continue;
    }

    if (auto load = dyn_cast<POP::CompilerGlobalLoadOp>(op)) {
      auto found = stores.find(load.getName());
      if (found == stores.end())
        continue;

      load.replaceAllUsesWith(found->getSecond().first.getValue());
      found->getSecond().second = true;
      load.erase();
    }
  }

  // Clean up all the stores that we were able to elide.
  for (auto [_, store] : stores)
    if (store.second)
      store.first.erase();
}
