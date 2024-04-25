//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "NVPTXAdapter.h"

#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"

using namespace M;
using namespace M::DebugInfo;

/// NVPTX does not support variables that have more than one location. This
/// means we cannot have a variable that has limited lifetime. Remove KillOps so
/// that instead of emitting multiple llvm DbgValueOps, we just emit a single
/// DbgDeclareOp when the variable does not change location.
static void removeDebugKills(mlir::ModuleOp module) {
  module->walk([](DebugInfo::KillOp kill) { kill->erase(); });
}

TargetAdapter DebugInfo::getNVPTXAdapter() {
  return TargetAdapter{removeDebugKills, convertDbgValueToDeclare};
}
