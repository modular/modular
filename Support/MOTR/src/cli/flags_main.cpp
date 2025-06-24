//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/motr.h"
#include <cassert>
#include <cerrno>
#include <list>
#include <string_view>

namespace Flags = M::motr::Flags;

using Duration = M::motr::Time::Duration;
using Timestamp = M::motr::Time::Timestamp;

int flagsMain(int argc, char **argv) {
  MOTR_Trace(flagsMain);

  auto parseError = [&](const std::string &msg = "") {
    MOTR_LOG("Error: {}", msg);
    MOTR_LOG("Usage: motr flags [get,set, getTimeout, setTimeout] [flagName] "
             "[value]",
             "");
    return 1;
  };

  std::list<std::string_view> args;
  for (int i = 0; i < argc; ++i)
    args.emplace_back(argv[i]);

  assert(args.size() >= 2);
  args.pop_front(); // remove program name
  assert(args.front() == "flags");
  args.pop_front();

  if (args.empty())
    return parseError("specify get or set");

  auto cmd = args.front();
  bool isSet = cmd == "set";
  bool isGet = cmd == "get";
  bool isGetTimeout = cmd == "getTimeout";
  bool isSetTimeout = cmd == "setTimeout";
  bool isReset = cmd == "reset";
  args.pop_front();

  if (!isSet && !isGet && !isGetTimeout && !isSetTimeout && !isReset)
    return parseError(
        "specify sub-command: reset, get, set, getTimeout, or setTimeout");

  if (isReset) {
    Flags::Manager::resetSharedMemory();
    MOTR_LOG("flags shared memory reset done.", "");
    return 0;
  }

  if (args.empty())
    return parseError("specify flagName");

  std::string flagName{args.front()}; // need string for Flag constructor
  args.pop_front();

  uint64_t value{};

  if (isSet || isSetTimeout) {
    if (args.empty())
      return parseError("specify value");

    std::string valueStr{args.front()}; // need string for c_str()
    args.pop_front();

    char *end;
    errno = 0;
    value = strtoull(valueStr.c_str(), &end, 10);
    if (errno != 0 || end != valueStr.c_str() + valueStr.length())
      return parseError(fmt::format("cannot parse value: {}", valueStr));
  }

  if (!args.empty())
    return parseError("too many arguments");

  if (isSet) {
    Flags::Flag setflag(flagName, value);
    uint64_t prev = setflag.set(value);
    MOTR_LOG("{}: {} -> {}", flagName, prev, value);
  }

  if (isSetTimeout) {
    Flags::Flag setflag(flagName);
    Duration timeout = Duration::fromMilliseconds(value);
    uint64_t prev = setflag.setGetTimeout(timeout.v);
    std::string timeoutStr = timeout.toString();
    std::string prevTimeoutStr = Duration::fromMilliseconds(prev).toString();
    MOTR_LOG("{}.timeout: {} ({}) -> {} ({})", flagName, prevTimeoutStr, prev,
             timeoutStr, value);
  }

  Flags::Flag getflag(flagName);
  MOTR_LOG("{}.hash=##{:016x}", flagName, getflag.hash());
  MOTR_LOG("{}.timeout={}", flagName, getflag.getGetTimeout());
  MOTR_LOG("{}.valid={}", flagName, getflag.valid());
  bool initialized = getflag.initialized();
  MOTR_LOG("{}.initialized={}", flagName, initialized);
  if (initialized) {
    auto value = getflag.getNoWait();
    MOTR_LOG("{}.value={}", flagName, value);
  } else {
    MOTR_LOG("{}.value={}", flagName, "<uninitialized>");
  }
  return 0;
}
