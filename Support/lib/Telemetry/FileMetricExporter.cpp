//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Exporters/FileMetricExporter.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "opentelemetry/sdk/common/exporter_utils.h"
#include "opentelemetry/sdk/metrics/export/metric_producer.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DebugLog.h"
#include <ios>

#define DEBUG_TYPE "telemetry-context"

using namespace M::Telemetry::Exporter;

opentelemetry::sdk::common::ExportResult FileMetricExporter::Export(
    const opentelemetry::sdk::metrics::ResourceMetrics &data) noexcept {

  if (filePath.empty())
    return opentelemetry::sdk::common::ExportResult::kSuccess;

  // Export data to outputStream using OTel's OStreamMetricExporter.
  ostreamExporter.Export(data);

  // If there is nothing to write, return.
  if (outputStream.tellp() == 0)
    return opentelemetry::sdk::common::ExportResult::kSuccess;

  // Flush the stream to a file.
  auto err = appendFileUnderLock(filePath, [&](llvm::raw_ostream &os) {
    // Do the stream manipulation inside the atomic region - other things may
    // try to write during this, and we need to hold the lock.
    os << outputStream.str();
    // Seek back to the beginning.
    outputStream.seekp(0, std::ios_base::beg);
  });
  if (err.isError()) {
    LDBG() << err.getError();
    return opentelemetry::sdk::common::ExportResult::kFailure;
  }

  return opentelemetry::sdk::common::ExportResult::kSuccess;
}
