//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_SUPPORT_STATSREPORT_H
#define TENSORFLOW_SUPPORT_STATSREPORT_H

#include "LLCL/Runtime/Runtime.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace M::TF {

/// A helper API that can collect statistics about ops lowered vs. fallback
/// ops during an external framework translation pass. For the fallback ops,
/// counts for each separate signature are gathered as well.
///
/// The helper also has the functionality to record these statistics to a
/// configurable report file.
struct StatsReport {
  explicit StatsReport(llvm::StringRef name)
      : name(name), numTotalOps(0), numFallbackOps(0) {
    ownedRuntime = LLCL::createRuntimeIfNeeded();
    ownedRuntime->emplaceContextIfMissing<Telemetry::TelemetryContext>();
  }

  StatsReport() : StatsReport("") {}

  /// Record one instance of an input op.
  void countOp(mlir::Operation &op);
  /// Record one instance of an op that was sent to fallback.
  void countFallback(mlir::Operation &op);
  /// Write the statistics to file.
  void writeToFile();

  // Emits telemetry info on the ops collected
  void emitTelemetry();

private:
  std::string name;
  size_t numTotalOps;
  size_t numFallbackOps;
  ConditionallyOwnedPointer<LLCL::Runtime> ownedRuntime;
  llvm::StringMap<size_t> fallbackHistogram;
  llvm::StringMap<size_t> opHistogram{};
};

} // namespace M::TF

#endif // TENSORFLOW_SUPPORT_STATSREPORT_H
