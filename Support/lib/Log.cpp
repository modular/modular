//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/Configuration.h"
#include "Support/Log.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>

#include <fmt/chrono.h>

namespace {
struct LogFormatState {
  bool initialized = false;
  bool useEnhancedFormat = true;
  bool showTimeStamp = true;
  bool showColors = true;
  bool showLogLevel = true;
  bool useIsoTimestamps = false;
  bool showMicroseconds = false;
  bool emitJSON = false;
}; // struct LogFormatState

} // namespace

namespace M::Log {

namespace ConfigEntry {
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

static void setLogLevelFromString(const llvm::StringRef level) {
  setLogLevel(parseLogLevelFromString(level));
}

static LogFormatState &getLogFormatState() {
  static LogFormatState state = []() {
    LogFormatState state;
    state.initialized = true;

    auto cfgOr = Config::open();
    if (cfgOr.isError())
      return state; // Use defaults if we can't read config

    using namespace ConfigEntry;
    auto cfg = cfgOr.takeValue();
    state.useEnhancedFormat = !cfg.getValueAsBool(LOG_NO_ENHANCED, false);
    state.showTimeStamp = !cfg.getValueAsBool(LOG_NO_TIMESTAMP, false);
    state.useIsoTimestamps = cfg.getValueAsBool(LOG_ISO_TIME, false);
    state.showMicroseconds = cfg.getValueAsBool(LOG_MICROSECONDS, false);
    state.emitJSON = cfg.getValueAsBool(LOG_JSON, false);

    // Respect the standard NO_COLOR env var, any value (even empty) disables
    // color.
    state.showColors = !llvm::sys::Process::GetEnv("NO_COLOR").has_value();

    // Check if the terminal does not support colors.
    if (state.showColors) {
      auto term = llvm::sys::Process::GetEnv("TERM");
      if ((term && *term == "dumb") ||
          !llvm::sys::Process::StandardOutIsDisplayed())
        state.showColors = false;
    }
    return state;
  }();
  return state;
}

static llvm::raw_ostream::Colors getLogLevelColor(LogLevel level) {
  using enum llvm::raw_ostream::Colors;
  if (!getLogFormatState().showColors)
    return RESET;
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

namespace {
std::atomic<Log::LogLevel> logLevel{Log::LogLevel::WARN};
}

// Initialize the log level from environment variable.
static void initLogLevel() {
  auto cfgOr = Config::open();
  if (cfgOr.isError()) {
    setLogLevel(LogLevel::WARN);
    return;
  }

  setLogLevelFromString(cfgOr.takeValue().getValue(ConfigEntry::LOG_LEVEL));
}

void setLogLevel(LogLevel level) {
  logLevel.store(level, std::memory_order::release);
}

LogLevel getLogLevel() {
  [[maybe_unused]] static bool initialized = []() -> bool {
    initLogLevel();
    return true;
  }();
  return logLevel.load(std::memory_order::acquire);
}

// Returns us in string form with a leading '.' and zero-padded to six digits
static llvm::SmallString<8> formatMicroseconds(
    std::chrono::time_point<std::chrono::system_clock> timePoint) {
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
static llvm::SmallString<32>
buildISOFormatString(std::chrono::time_point<std::chrono::system_clock> now,
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

static llvm::SmallString<32> buildTimestampString() {
  llvm::SmallString<32> result;
  const auto &state = getLogFormatState();
  if (!state.showTimeStamp)
    return result;

  auto now = std::chrono::system_clock::now();
  if (state.useIsoTimestamps)
    return buildISOFormatString(now, state.showMicroseconds);

  // Simple format: 16:21:14
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(now));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> time;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  time.resize(nChars, '\0');
  auto formatResult = fmt::format_to_n(time.data(), nChars, "{:%H:%M:%S}", utc);
  time.resize(formatResult.size);
  result += time;
  if (state.showMicroseconds)
    result += formatMicroseconds(now);
  return result;
}

static llvm::SmallString<128> buildLogPrefix(LogLevel level) {
  using enum llvm::raw_ostream::Colors;
  const auto &state = getLogFormatState();
  llvm::SmallString<128> prefix;

  if (state.showTimeStamp) {
    prefix += "[";
    prefix += buildTimestampString();
    prefix += "] ";
  }

  if (state.showLogLevel) {
    if (state.showColors)
      prefix += llvm::sys::Process::OutputColor(
          static_cast<char>(getLogLevelColor(level)), /*bold=*/false,
          /*bg=*/false);
    prefix += "[";
    prefix += getLogLevelPrefix(level);
    prefix += "] ";
    if (state.showColors)
      prefix += llvm::sys::Process::ResetColor();
  }

  return prefix;
}

static llvm::SmallString<512> buildJSONLogLine(LogLevel level,
                                               llvm::StringRef msg) {
  // Always use full ISO 8601 with microseconds in JSON mode regardless of
  // the showTimeStamp / useIsoTimestamps / showMicroseconds config flags.
  auto now = std::chrono::system_clock::now();
  auto timestamp = buildISOFormatString(now, /*includeMicroseconds=*/true);

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

static void writeStringToLogFileOrStdout(llvm::StringRef msg) {
  static std::mutex logFileMutex;
  static auto ostream = []() {
    auto cfgOr = Config::open();
    auto logFileName = cfgOr.isError()
                           ? llvm::StringRef()
                           : cfgOr.get().getValue(ConfigEntry::LOG_FILE);
    llvm::raw_ostream *stream = &llvm::outs();
    if (!logFileName.empty()) {
      std::error_code ec;
      static llvm::raw_fd_ostream fdStream(
          logFileName, ec, llvm::sys::fs::CD_OpenAlways,
          llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Append);
      if (!ec) {
        stream = &fdStream;
      }
    }
    return stream;
  }();

  std::lock_guard<std::mutex> lock(logFileMutex);
  ostream->write(msg.begin(), msg.size());
  ostream->write('\n');
  ostream->flush();
}

void logWrite(LogLevel level, llvm::StringRef msg) {
  if (getLogFormatState().emitJSON)
    return writeStringToLogFileOrStdout(buildJSONLogLine(level, msg));
  if (!getLogFormatState().useEnhancedFormat)
    return writeStringToLogFileOrStdout(msg);

  auto enhanced = buildLogPrefix(level);
  enhanced += msg;
  writeStringToLogFileOrStdout(enhanced);
}

} // namespace M::Log
