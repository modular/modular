//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H
#define SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H

#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/exporters/ostream/log_record_exporter.h"
#include "opentelemetry/sdk/logs/recordable.h"
#include <filesystem>
#include <sstream>
#endif // MODULAR_ENABLE_TELEMETRY

namespace M::Telemetry::Exporter {

#ifdef MODULAR_ENABLE_TELEMETRY

/// The FileLogExporter exports log data to a file, leveraging
/// OTel's OStreamLogRecordExporter.
class FileLogExporter : public opentelemetry::sdk::logs::LogRecordExporter {
public:
  explicit FileLogExporter(std::filesystem::path filePath)
      : filePath(std::move(filePath)), ostreamExporter(outputStream) {}

  virtual ~FileLogExporter() = default;

  std::unique_ptr<opentelemetry::sdk::logs::Recordable>
  MakeRecordable() noexcept override {
    return ostreamExporter.MakeRecordable();
  }

  /// Exports a span of logs sent from the processor to a file.
  opentelemetry::sdk::common::ExportResult
  Export(const opentelemetry::nostd::span<std::unique_ptr<
             opentelemetry::sdk::logs::Recordable>> &records) noexcept override;

  /// Force flush the exporter.
  bool ForceFlush(std::chrono::microseconds timeout =
                      std::chrono::microseconds::max()) noexcept override {
    return ostreamExporter.ForceFlush(timeout);
  }

  /// Shut down the exporter, with optional timeout.
  bool Shutdown(std::chrono::microseconds timeout =
                    std::chrono::microseconds::max()) noexcept override {
    return ostreamExporter.Shutdown(timeout);
  }

private:
  /// Logs are exported to this file.
  std::filesystem::path filePath;
  /// Buffer OTel's outputs in a string and flush it atomically to a file every
  /// time we export.
  std::stringstream outputStream;
  /// Delegate printing of telemetry data to OTel's OStreamLogRecordExporter.
  opentelemetry::exporter::logs::OStreamLogRecordExporter ostreamExporter;
};

#endif // MODULAR_ENABLE_TELEMETRY

} // namespace M::Telemetry::Exporter

#endif // SUPPORT_TELEMETRY_EXPORTERS_FILELOGEXPORTER_H
