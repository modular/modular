//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_LOGS_H
#define SUPPORT_TELEMETRY_LOGS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/ForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/logs/event_logger.h"
#include "opentelemetry/logs/severity.h"
#endif // MODULAR_ENABLE_TELEMETRY
#include <unordered_map>
#include <variant>

namespace M::Telemetry::Logs {

/// Severity levels for logs.
/// See
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber

#ifdef MODULAR_ENABLE_TELEMETRY

typedef opentelemetry::logs::Severity Severity;

using AttributeValue = opentelemetry::common::AttributeValue;

#else

// If telemetry is disabled, this enum is only required for the NOOP API.
// Note: not including all of OTel's severity numbers here, only the ones
// we are going to use.
enum class Severity : uint8_t {
  kInvalid,
  kTrace,
  kDebug,
  kInfo,
  kWarn,
  kError,
  kFatal
};

using AttributeValue = std::variant<bool, int32_t, int64_t, uint32_t, double,
                                    const char *, std::string, uint64_t>;

#endif // MODULAR_ENABLE_TELEMETRY

/// A Logger to emit logs. Logger's methods are thread-safe.
/// Usage examples:
/// - logger->getError("my-event") << "my log error message";
/// - logger->emitEvent("my-event", Severity::kError, "my log error message");
/// - logger->emitEvent("billing", Severity::kInfo);
class Logger : public std::enable_shared_from_this<Logger> {
public:
  /// Emit event with given name, severity, body and attributes.
  void emitEvent(StringRef eventName, Severity severity,
                 M::Telemetry::Level level, StringRef body,
                 const llvm::StringMap<AttributeValue> &attributes) {
#ifdef MODULAR_ENABLE_TELEMETRY
    if (eventEnabled(level)) {
      // Convert the attributes to unordered_map to pass to OTel.
      std::unordered_map<std::string, AttributeValue> attrs;
      for (auto &attr : attributes) {
        std::visit([&](auto v) { attrs[attr.first().str()] = v; }, attr.second);
      }
      logger->EmitEvent(eventName,
                        static_cast<opentelemetry::logs::Severity>(severity),
                        body, attrs);
    }
#endif
  }
  void emitEvent(StringRef eventName, Severity severity,
                 M::Telemetry::Level level) {
    llvm::StringMap<AttributeValue> attrs;
    return emitEvent(eventName, severity, level, "", attrs);
  }
  void emitEvent(StringRef eventName, Severity severity,
                 M::Telemetry::Level level, StringRef body) {
    llvm::StringMap<AttributeValue> attrs;
    return emitEvent(eventName, severity, level, body, attrs);
  }

  /// Returns true if an event will be emitted based on its level and the
  /// configured telemetry level.
  bool eventEnabled(Level eventLevel) const {
    return eventLevel <= telemetryLevel;
  }

  /// LogStream is a llvm::raw_ostream wrapper around Logger::emitEvent().
  /// Logger has convenience methods that return a LogStream with a preset
  /// severity level (e.g. Logger::getWarn() returns a LogStream with severity
  /// of warn). LogStream is implemented by having a subclass of raw_ostream
  /// where event emission happens when the stream goes out of scope. The stream
  /// needs a pointer to the logger to emit the event. The pointer is shared and
  /// obtained from logger with shared_from_this() to avoid situations where the
  /// stream might outlive the logger.
  struct LogStream : public llvm::raw_string_ostream {

    virtual ~LogStream() {
      logger->emitEvent(eventName, severity, level, body, attributes);
    }

    /// Provide an explicit overload for strings that escapes special
    /// characters.
    friend LogStream &operator<<(LogStream &os, StringRef str) {
      os.write_escaped(str);
      return os;
    }

  private:
    friend class Logger;

    LogStream(const Twine &eventName, Severity severity,
              M::Telemetry::Level level,
              const llvm::StringMap<AttributeValue> &attrs,
              std::shared_ptr<Logger> logger)
        : raw_string_ostream(body), eventName(eventName.str()),
          severity(severity), level(level), logger(logger) {
      for (auto &attr : attrs) {
        std::visit([&](auto v) { attributes[attr.first().str()] = v; },
                   attr.second);
      }
    }

    std::string body;
    std::string eventName;
    Severity severity;
    M::Telemetry::Level level;
    llvm::StringMap<AttributeValue> attributes;
    // TODO: timestamp.
    std::shared_ptr<Logger> logger;
  };

  /// Get raw_ostream to write a log with a severity of trace.
  LogStream getTrace(const Twine &eventName, M::Telemetry::Level level,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kTrace, level, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of debug.
  LogStream getDebug(const Twine &eventName, M::Telemetry::Level level,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kDebug, level, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of info.
  LogStream getInfo(const Twine &eventName, M::Telemetry::Level level,
                    const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kInfo, level, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of warn.
  LogStream getWarn(const Twine &eventName, M::Telemetry::Level level,
                    const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kWarn, level, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of error.
  LogStream getError(const Twine &eventName, M::Telemetry::Level level,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kError, level, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of fatal.
  LogStream getFatal(const Twine &eventName, M::Telemetry::Level level,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kFatal, level, attributes,
                     shared_from_this());
  }

private:
  friend class M::Telemetry::TelemetryContext;

#ifdef MODULAR_ENABLE_TELEMETRY
  Logger(std::shared_ptr<opentelemetry::logs::EventLogger> logger,
         M::Telemetry::Level level)
      : logger(logger), telemetryLevel(level) {}

  std::shared_ptr<opentelemetry::logs::EventLogger> logger;
#else
  Logger() {}
#endif // MODULAR_ENABLE_TELEMETRY

  // Configured level for Telemetry.
  M::Telemetry::Level telemetryLevel;
};

} // namespace M::Telemetry::Logs

#endif // SUPPORT_TELEMETRY_LOGS_H
