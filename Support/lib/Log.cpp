//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

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
}; // struct LogFormatState

} // namespace

namespace M::Log {

namespace EnvVar {
static constexpr llvm::StringLiteral LOG_FILE = "MODULAR_LOG_FILE";
static constexpr llvm::StringLiteral LOG_ISO_TIME = "MODULAR_LOG_ISO_TIME";
static constexpr llvm::StringLiteral LOG_LEVEL = "MODULAR_LOG_LEVEL";
static constexpr llvm::StringLiteral LOG_MICROSECONDS =
    "MODULAR_LOG_MICROSECONDS";
static constexpr llvm::StringLiteral LOG_NO_ENHANCED =
    "MODULAR_LOG_NO_ENHANCED";
static constexpr llvm::StringLiteral LOG_NO_TIMESTAMP =
    "MODULAR_LOG_NO_TIMESTAMP";
} // namespace EnvVar

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

    // Helper lambda to check if an env var equals a value of "1".
    auto envIsSet = [](llvm::StringLiteral envVar) -> bool {
      auto env = llvm::sys::Process::GetEnv(envVar.str());
      return env && *env == "1";
    };

    state.useEnhancedFormat = !envIsSet(EnvVar::LOG_NO_ENHANCED);
    state.showTimeStamp = !envIsSet(EnvVar::LOG_NO_TIMESTAMP);
    state.useIsoTimestamps = envIsSet(EnvVar::LOG_ISO_TIME);
    state.showMicroseconds = envIsSet(EnvVar::LOG_MICROSECONDS);

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
  auto env = llvm::sys::Process::GetEnv(EnvVar::LOG_LEVEL.str());
  if (!env) {
    setLogLevel(LogLevel::WARN);
    return;
  }

  setLogLevelFromString(*env);
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

static llvm::SmallString<32> buildTimestampString() {

  llvm::SmallString<32> result;
  const auto &state = getLogFormatState();
  if (!state.showTimeStamp)
    return result;

  auto now = std::chrono::system_clock::now();
  std::tm utc = fmt::gmtime(std::chrono::system_clock::to_time_t(now));
  constexpr size_t nChars = 32;
  llvm::SmallString<nChars> time;
  // Pre-size the buffer before format_to_n. Sizing after would call append()
  // from size 0, which overwrites the just-formatted data with zeros.
  time.resize(nChars, '\0');
  // The format message has to be constant so we emit two separate format_to_n
  // calls based on the useIsoTimestamps flag.
  fmt::format_to_n_result<char *> formatResult;
  if (state.useIsoTimestamps) // ISO 8601 format: 2026-12-25T12:00:00.123450Z
    formatResult =
        fmt::format_to_n(time.data(), nChars, "{:%Y-%m-%dT%H:%M:%S}", utc);
  else // Simple format: 16:21:14
    formatResult = fmt::format_to_n(time.data(), nChars, "{:%H:%M:%S}", utc);
  time.resize(formatResult.size);
  result += time;
  if (state.showMicroseconds)
    result += formatMicroseconds(now);
  if (state.useIsoTimestamps)
    result += "Z";
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

static void writeStringToLogFileOrStdout(llvm::StringRef msg) {
  static auto logFileName = llvm::sys::Process::GetEnv(EnvVar::LOG_FILE.str());
  static std::mutex logFileMutex;
  static auto ostream = [&]() {
    llvm::raw_ostream *stream = &llvm::outs();
    if (logFileName) {
      std::error_code ec;
      static llvm::raw_fd_ostream fdStream(
          *logFileName, ec, llvm::sys::fs::CD_OpenAlways,
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
  if (!getLogFormatState().useEnhancedFormat)
    return writeStringToLogFileOrStdout(msg);

  auto enhanced = buildLogPrefix(level);
  enhanced += msg;
  writeStringToLogFileOrStdout(enhanced);
}

} // namespace M::Log
