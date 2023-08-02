//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifdef MODULAR_ENABLE_TELEMETRY

#include "LLCL/Support/Telemetry/Exporters/FileLogExporter.h"

#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "telemetry-context"

using namespace M::LLCL::Telemetry::Exporter;

opentelemetry::sdk::common::ExportResult
FileLogExporter::Export(const opentelemetry::nostd::span<
                        std::unique_ptr<opentelemetry::sdk::logs::Recordable>>
                            &records) noexcept {

  if (filePath.empty())
    return opentelemetry::sdk::common::ExportResult::kSuccess;

  // Export data to outputStream using OTel's OStreamLogRecordExporter.
  ostreamExporter.Export(records);

  // Flush the stream to a file.
  auto err = appendFileUnderLock(filePath, [&](llvm::raw_ostream &os) {
    // Do the stream manipulation inside the atomic region - other things may
    // try to write during this, and we need to hold the lock.
    os << outputStream.str();
    // Seek back to the beginning.
    outputStream.seekp(0, std::ios_base::beg);
  });
  if (err.isError()) {
    LLVM_DEBUG(llvm::dbgs() << err.getError());
    return opentelemetry::sdk::common::ExportResult::kFailure;
  }

  return opentelemetry::sdk::common::ExportResult::kSuccess;
}

#endif // MODULAR_ENABLE_TELEMETRY
