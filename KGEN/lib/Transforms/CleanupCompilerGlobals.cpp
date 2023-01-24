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
  DenseMap<StringAttr, POP::CompilerGlobalStoreOp> stores;
  // First we collect all the store ops.
  func.walk([&](POP::CompilerGlobalStoreOp store) {
    stores.try_emplace(store.getNameAttr(), store);
  });

  // Then we can walk all the load ops.
  func.walk([&](POP::CompilerGlobalLoadOp load) {
    auto found = stores.find(load.getNameAttr());
    if (found == stores.end())
      return;

    load.replaceAllUsesWith(found->getSecond().getValue());
    load.erase();
  });

  // Clean up all the stores we were able to elide.
  for (auto [_, store] : stores)
    store.erase();
}
