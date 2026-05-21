//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Log.h"
#include "Support/Configuration.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <fmt/args.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>

namespace M::Log {

namespace ConfigEntry {
static constexpr llvm::StringLiteral LOG_STDOUT = "log.stdout";
static constexpr llvm::StringLiteral LOG_FILE = "log.file";
static constexpr llvm::StringLiteral LOG_ISO_TIME = "log.iso_time";
static constexpr llvm::StringLiteral LOG_LEVEL = "log.level";
static constexpr llvm::StringLiteral LOG_MICROSECONDS = "log.microseconds";
static constexpr llvm::StringLiteral LOG_NO_ENHANCED = "log.no_enhanced";
static constexpr llvm::StringLiteral LOG_NO_TIMESTAMP = "log.no_timestamp";
static constexpr llvm::StringLiteral LOG_JSON = "log.json";
} // namespace ConfigEntry

static LogLevel parseLogLevelFromString(llvm::StringRef levelStr) {
  if (levelStr == "0")
    return LogLevel::DEBUG;
  if (levelStr == "1")
    return LogLevel::INFO;
  if (levelStr == "2")
    return LogLevel::WARN;
  if (levelStr == "3")
    return LogLevel::ERROR;
  if (levelStr == "4")
    return LogLevel::FATAL;
  if (levelStr.equals_insensitive("DEBUG"))
    return LogLevel::DEBUG;
  if (levelStr.equals_insensitive("INFO"))
    return LogLevel::INFO;
  if (levelStr.equals_insensitive("WARN"))
    return LogLevel::WARN;
  if (levelStr.equals_insensitive("ERROR"))
    return LogLevel::ERROR;
  if (levelStr.equals_insensitive("FATAL"))
    return LogLevel::FATAL;

  return LogLevel::WARN;
}

static llvm::raw_ostream::Colors getLogLevelColor(LogLevel level) {
  using enum llvm::raw_ostream::Colors;
  switch (level) {
  case LogLevel::DEBUG:
    return BRIGHT_BLACK;
  case LogLevel::INFO:
    return BRIGHT_CYAN;
  case LogLevel::WARN:
    return BRIGHT_YELLOW;
  case LogLevel::ERROR:
    return BRIGHT_RED;
  case LogLevel::FATAL:
    return RED;
  }
}

static llvm::StringLiteral getLogLevelPrefix(LogLevel level) {
  switch (level) {
  case LogLevel::DEBUG:
    return " DBG";
  case LogLevel::INFO:
    return "INFO";
  case LogLevel::WARN:
    return "WARN";
  case LogLevel::ERROR:
    return " ERR";
  case LogLevel::FATAL:
    return "FATL";
  }
}

// Returns us in string form with a leading '.' and zero-padded to six digits
static llvm::SmallString<8> formatMicroseconds(LogRecord::Timestamp timePoint) {
  llvm::SmallString<8> result;
  // Only count microseconds by modulo'ing by 1 million
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                    timePoint.time_since_epoch()) %
                1'000'000;
  // us are 6 digits long, plus the single '.'
  result.resize(7, 0);
  fmt::format_to_n(result.data(), result.size(), ".{:06}", micros.count());
  return result;
}

// Returns a full ISO 8601 UTC timestamp, e.g. "2026-12-25T12:00:00.123456Z".
// The result is always 20 chars (without microseconds) or 27 chars (with).
// SmallString<32> is sized to hold either with room to spare.
static llvm::SmallString<32> buildISOFormatString(LogRecord::Timestamp now,
                                                  bool includeMicroseconds) {
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(now));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> result;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  result.resize(nChars, '\0');
  auto fmtResult =
      fmt::format_to_n(result.data(), nChars, "{:%Y-%m-%dT%H:%M:%S}", utc);
  result.resize(fmtResult.size);
  if (includeMicroseconds)
    result += formatMicroseconds(now);
  result += "Z";
  return result;
}

static llvm::SmallString<512>
buildJSONLogLine(LogLevel level, llvm::StringRef msg, LogRecord::Timestamp ts) {
  // Always use full ISO 8601 with microseconds in JSON mode regardless of
  // the showTimeStamp / useIsoTimestamps / showMicroseconds config flags.
  auto timestamp = buildISOFormatString(ts, /*includeMicroseconds=*/true);

  llvm::SmallString<512> jsonLogLine;
  llvm::raw_svector_ostream svOstream(jsonLogLine);
  llvm::json::OStream json(svOstream);
  json.object([&] {
    json.attribute("timestamp", timestamp);
    json.attribute("level", getLogLevelPrefix(level).trim());
    json.attribute("message", msg);
  });
  return jsonLogLine;
}

static std::string renderRecord(const LogRecord &record) {
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  store.reserve(record.argCount, /*new_cap_named=*/0);
  for (const auto &arg : std::span(record.args.data(), record.argCount)) {
    switch (arg.tag) {
    case LogArg::Type::Bool:
      store.push_back(arg.data.b);
      break;
    case LogArg::Type::Int64:
      store.push_back(arg.data.i64);
      break;
    case LogArg::Type::UInt64:
      store.push_back(arg.data.ui64);
      break;
    case LogArg::Type::Fp32:
      store.push_back(arg.data.fp32);
      break;
    case LogArg::Type::Fp64:
      store.push_back(arg.data.fp64);
      break;
    case LogArg::Type::SmallString:
      store.push_back(fmt::string_view(
          arg.data.ssoStr.data(),
          strnlen(arg.data.ssoStr.data(), arg.data.ssoStr.size())));
      break;
    case LogArg::Type::String:
      store.push_back(fmt::string_view(arg.data.str.ptr, arg.data.str.len));
      break;
    case LogArg::Type::Pointer:
      store.push_back(fmt::ptr(arg.data.ptr));
      break;
    }
  }
  return fmt::vformat(record.fmtString, store);
}

class Sink {
public:
  virtual ~Sink() = default;
  virtual void write(llvm::StringRef msg) = 0;
};

class FileSink : public Sink {
  std::error_code ec;
  llvm::raw_fd_ostream ostream;
  std::mutex outputMutex;

  FileSink(llvm::StringRef path)
      : ostream{path, ec, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::FA_Write,
                llvm::sys::fs::OF_Append} {}

public:
  static llvm::ErrorOr<std::unique_ptr<FileSink>> create(llvm::StringRef path) {
    auto fileSink = std::unique_ptr<FileSink>(new FileSink(path));
    if (fileSink->ec)
      return fileSink->ec;
    return fileSink;
  }

  void write(llvm::StringRef msg) override {
    std::lock_guard<std::mutex> lock(outputMutex);
    ostream.write(msg.begin(), msg.size());
    ostream.write('\n');
    ostream.flush();
  }
};

class StdoutSink : public Sink {
  std::mutex outputMutex;

  void write(llvm::StringRef msg) override {
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::outs() << msg << "\n";
    llvm::outs().flush();
  }
};

Logger::Logger() {
  // Respect the standard NO_COLOR env var, any value (even empty) disables
  // color - does not depend on config object.
  formatState.showColors = !llvm::sys::Process::GetEnv("NO_COLOR").has_value();

  // Check if the terminal does not support colors.
  if (formatState.showColors) {
    auto term = llvm::sys::Process::GetEnv("TERM");
    if ((term && *term == "dumb") ||
        !llvm::sys::Process::StandardOutIsDisplayed())
      formatState.showColors = false;
  }

  auto cfgOr = Config::open();
  if (cfgOr.isError()) {
    // If we can't read the config, the default is to log to stdout.
    sinks.push_back(std::make_unique<StdoutSink>());
    return;
  }
  auto cfg = cfgOr.takeValue();
  using namespace ConfigEntry;
  formatState.useEnhancedFormat = !cfg.getValueAsBool(LOG_NO_ENHANCED, false);
  formatState.showTimeStamp = !cfg.getValueAsBool(LOG_NO_TIMESTAMP, false);
  formatState.useIsoTimestamps = cfg.getValueAsBool(LOG_ISO_TIME, false);
  formatState.showMicroseconds = cfg.getValueAsBool(LOG_MICROSECONDS, false);
  formatState.emitJSON = cfg.getValueAsBool(LOG_JSON, false);
  auto logToStdout = cfg.getValueAsBool(LOG_STDOUT, true);

  this->setLogLevel(
      parseLogLevelFromString(cfg.getValueOr(ConfigEntry::LOG_LEVEL, "WARN")));

  // If stdout logging is requested or if no log file present, log to stdout.
  // The stdout variable is default-true, but can be overridden.
  auto logFilePath = cfg.getValueOr(ConfigEntry::LOG_FILE, "");
  if (logToStdout)
    sinks.push_back(std::make_unique<StdoutSink>());
  if (!logFilePath.empty()) {
    auto fileSinkOrErr = FileSink::create(logFilePath);
    if (fileSinkOrErr)
      sinks.push_back(std::move(*fileSinkOrErr));
    else
      llvm::errs() << "Failed to open log file '" << logFilePath
                   << "': " << fileSinkOrErr.getError().message()
                   << (logToStdout ? "\nLog messages only going to stdout.\n"
                                   : "\nNo log messages will be emitted.\n");
  }
}

Logger::~Logger() = default;

void Logger::log(LogRecord record) {
  auto msg = renderRecord(record);
  auto level = record.level;
  llvm::SmallString<512> enhancedOrJSONMsg;
  if (formatState.emitJSON) {
    enhancedOrJSONMsg = buildJSONLogLine(level, msg, record.timestamp);
  } else if (formatState.useEnhancedFormat) {
    enhancedOrJSONMsg = buildLogPrefix(level, record.timestamp);
    enhancedOrJSONMsg += msg;
  } else {
    enhancedOrJSONMsg = msg;
  }

  for (const auto &sink : sinks) {
    sink->write(enhancedOrJSONMsg);
  }
}

// Depends on logger state to access formatting options
// TODO(dmcbain): change how format options are used to move this out-of-line
llvm::SmallString<32> Logger::buildTimestampString(LogRecord::Timestamp ts) {
  llvm::SmallString<32> result;
  if (!formatState.showTimeStamp)
    return result;

  if (formatState.useIsoTimestamps)
    return buildISOFormatString(ts, formatState.showMicroseconds);

  // Simple format: 16:21:14
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(ts));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> time;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  time.resize(nChars, '\0');
  auto formatResult = fmt::format_to_n(time.data(), nChars, "{:%H:%M:%S}", utc);
  time.resize(formatResult.size);
  result += time;
  if (formatState.showMicroseconds)
    result += formatMicroseconds(ts);
  return result;
}

// TODO(dmcbain): change how format options are used to move this out-of-line
llvm::SmallString<128> Logger::buildLogPrefix(LogLevel level,
                                              LogRecord::Timestamp ts) {
  using enum llvm::raw_ostream::Colors;
  llvm::SmallString<128> prefix;

  if (formatState.showTimeStamp) {
    prefix += "[";
    prefix += buildTimestampString(ts);
    prefix += "] ";
  }

  if (formatState.showLogLevel) {
    if (formatState.showColors)
      prefix += llvm::sys::Process::OutputColor(
          static_cast<char>(getLogLevelColor(level)), /*bold=*/false,
          /*bg=*/false);
    prefix += "[";
    prefix += getLogLevelPrefix(level);
    prefix += "] ";
    if (formatState.showColors)
      prefix += llvm::sys::Process::ResetColor();
  }

  return prefix;
}

} // namespace M::Log
