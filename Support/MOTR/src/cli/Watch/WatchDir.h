//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef WATCH_DIR_H
#define WATCH_DIR_H

#include "motr/Common.h"

#include <chrono>
#include <string>
#include <vector>

#include "Config/ConfigFile.h"
#ifdef MOTR_PLATFORM_MACOS
#include <CoreServices/CoreServices.h>
#endif

namespace M::motr {
struct WatchDir;
struct Task;
} // namespace M::motr

struct M::motr::Task {
  Task(const M::motr::TaskConfig &config);

  M::motr::TaskConfig config;
  bool execute() const;
};

struct M::motr::WatchDir {

  WatchDir(const M::motr::WatchConfig &config);
  ~WatchDir();
  WatchDir(const WatchDir &) = delete;
  WatchDir &operator=(const WatchDir &) = delete;
  WatchDir(WatchDir &&) = delete;
  WatchDir &operator=(WatchDir &&) = delete;

  void start();
  void stop();

  void add_task(const std::string &task);

#ifndef MOTR_PLATFORM_MACOS
  using FSEventStreamRef = void *;
  using ConstFSEventStreamRef = const void *;
  using FSEventStreamEventFlags = unsigned;
  using FSEventStreamEventId = int;
#endif

  static void callback(ConstFSEventStreamRef streamRef,
                       void *clientCallBackInfo, size_t numEvents,
                       void *eventPaths,
                       const FSEventStreamEventFlags eventFlags[],
                       const FSEventStreamEventId eventIds[]);

  void execute_tasks();

  M::motr::WatchConfig config;
  std::vector<Task> tasks;

  FSEventStreamRef eventStream = nullptr;
  uint64_t timestampLastEvent = 0;
  void scheduleExecution();
  void checkPendingExecution();
};

#endif // WATCH_DIR_H
