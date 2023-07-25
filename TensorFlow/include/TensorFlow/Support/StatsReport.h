//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_SUPPORT_STATSREPORT_H
#define TENSORFLOW_SUPPORT_STATSREPORT_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Operation.h"

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
      : name(name), numTotalOps(0), numFallbackOps(0) {}

  StatsReport() : StatsReport("") {}

  /// Record one instance of an input op.
  void countOp();
  /// Record one instance of an op that was sent to fallback.
  void countFallback(mlir::Operation &op);
  /// Write the statistics to file.
  void writeToFile();

private:
  std::string name;
  size_t numTotalOps;
  size_t numFallbackOps;
  std::unordered_map<std::string, int> fallbackHistogram;
};

} // namespace M::TF

#endif // TENSORFLOW_SUPPORT_STATSREPORT_H
