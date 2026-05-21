//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/LogFFI.h"
#include "Support/Log.h"
#include "Support/SymbolExport.h"

#include <chrono>
#include <cstring>

using namespace M::Log;

static_assert(std::is_trivially_copyable_v<LogArg>);

MODULAR_EXPORT int64_t MLog_now(void) {
  return std::chrono::system_clock::now().time_since_epoch().count();
}

MODULAR_EXPORT uint8_t MLog_get_level(void) {
  return static_cast<uint8_t>(getDefaultLog().getLogLevel());
}

MODULAR_EXPORT void MLog_set_level(uint8_t level) {
  getDefaultLog().setLogLevel(static_cast<LogLevel>(level));
}

MODULAR_EXPORT void MLog_write(uint8_t level, int64_t timestamp,
                               const char *fmt, size_t fmtLen, const void *args,
                               uint8_t argCount) {
  auto &log = getDefaultLog();
  if (log.getLogLevel() > static_cast<LogLevel>(level))
    return;

  std::array<LogArg, LogRecord::maxArgs> argArr{};
  std::memcpy(argArr.data(), args, argCount * sizeof(LogArg));

  LogRecord::Timestamp ts{LogRecord::Timestamp::clock::duration(timestamp)};
  LogRecord record(ts, static_cast<LogLevel>(level), {fmt, fmtLen},
                   std::move(argArr), argCount);
  logWrite(log, std::move(record));
}
