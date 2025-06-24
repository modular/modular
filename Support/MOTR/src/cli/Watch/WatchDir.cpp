//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "WatchDir.h"
#include "motr/Log.h"
#include "motr/motr.h"
#include <cstdlib>
#include <iostream>
#include <sys/wait.h>
#include <thread>

#define FMT_HEADER_ONLY
// todo: remove this in favor of compiling .cpp
#include "fmt/format.h"

using namespace M;

motr::WatchDir::WatchDir(const motr::WatchConfig &config) : config(config) {
  auto &global_config = motr::ConfigFile::getSingleton();

  for (const auto &task_config_name : config.tasks) {
    auto it = global_config.tasks.find(task_config_name);
    if (it == global_config.tasks.end()) {
      MOTR_LOG("WatchDir::WatchDir task[{}] not found in global config",
               task_config_name);
      continue;
    }

    tasks.emplace_back(it->second);
  }

  MOTR_LOG("WatchDir::WatchDir path={} ntasks={}", config.path, tasks.size());
}

motr::WatchDir::~WatchDir() {}

void motr::WatchDir::start() {
  if (eventStream)
    return;

#if defined(MOTR_PLATFORM_MACOS)
  CFStringRef pathRef = CFStringCreateWithCString(nullptr, config.path.c_str(),
                                                  kCFStringEncodingUTF8);
  CFArrayRef pathsToWatch =
      CFArrayCreate(nullptr, (const void **)&pathRef, 1, nullptr);

  FSEventStreamContext context = {0, this, nullptr, nullptr, nullptr};

  eventStream = FSEventStreamCreate(
      nullptr, &WatchDir::callback, &context, pathsToWatch,
      kFSEventStreamEventIdSinceNow, 1.0,
      kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagWatchRoot);
#endif

#if defined(MOTR_PLATFORM_MACOS)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  FSEventStreamScheduleWithRunLoop(eventStream, CFRunLoopGetCurrent(),
                                   kCFRunLoopDefaultMode);
#pragma clang diagnostic pop

  FSEventStreamStart(eventStream);

  CFRelease(pathsToWatch);
  CFRelease(pathRef);
#endif

  timestampLastEvent = 0;
}

void motr::WatchDir::stop() {
  if (!eventStream)
    return;

#if defined(MOTR_PLATFORM_MACOS)
  FSEventStreamStop(eventStream);
  FSEventStreamInvalidate(eventStream);
  FSEventStreamRelease(eventStream);
#endif
  eventStream = nullptr;
}

static std::vector<std::string> getEventPaths(size_t numEvents,
                                              void *eventPaths) {
  std::vector<std::string> paths;
  return paths;
  paths.reserve(numEvents);

#if defined(MOTR_PLATFORM_MACOS)
  char pathBuffer[PATH_MAX];
  auto cfPaths = static_cast<CFArrayRef>(eventPaths);

  for (size_t i = 0; i < numEvents; i++) {
    CFStringRef pathRef = (CFStringRef)CFArrayGetValueAtIndex(cfPaths, i);
    if (CFStringGetCString(pathRef, pathBuffer, PATH_MAX,
                           kCFStringEncodingUTF8))
      paths.emplace_back(pathBuffer);
  }
#endif

  return paths;
}

void motr::WatchDir::callback(ConstFSEventStreamRef streamRef,
                              void *clientCallBackInfo, size_t numEvents,
                              void *eventPaths,
                              const FSEventStreamEventFlags eventFlags[],
                              const FSEventStreamEventId eventIds[]) {
#if defined(MOTR_PLATFORM_MACOS)
  WatchDir *watcher = static_cast<WatchDir *>(clientCallBackInfo);
  // MOTR_LOG("\n[watch] [{}] {}", watcher->config.name, watcher->config.path);
  // todo: inspect eventPaths and have better logic for deciding if we should
  // execute
  watcher->scheduleExecution();
#endif
}

void motr::WatchDir::scheduleExecution() {
  if (eventStream)
    timestampLastEvent = nowNanoSeconds();
}

void motr::WatchDir::checkPendingExecution() {
  if (timestampLastEvent == 0)
    return;

  auto now = nowNanoSeconds();
  auto elapsed = (now - timestampLastEvent) / 1e9;
  if (elapsed >= config.latency) {
    MOTR_LOG("\n[watch] [{}] {} changed {:0.2f}s ago, now executing tasks...",
             config.name, config.path, elapsed);
    execute_tasks();
    timestampLastEvent = 0;
  }
}

motr::Task::Task(const motr::TaskConfig &config) : config(config) {}

bool motr::Task::execute() const {
  bool isCommand = !config.command.empty();
  bool isMessage = config.message != motr::MessageType::None;
  assert(isCommand != isMessage);
  if (isCommand) {
    MOTR_LOG("Task[{}] execute command: {}", config.name, config.command);
    int status = std::system(config.command.c_str());
    int exit_code = WEXITSTATUS(status);
    MOTR_LOG("Task[{}] command exit status=0x{:04x}, exit_code={}", config.name,
             status, exit_code);
    return exit_code == 0;
  }

  if (isMessage) {
    MOTR_LOG("Task[{}] execute message: {}", config.name,
             motr::toString(config.message));
    {
      ServerOutbox::getQueue().debugPrint();
      EmitMessage<MessageType::None> msg{};
      msg.msg.type = config.message;
      msg.send();
    }
    MOTR_LOG("EmitMessage done", "");
    // todo: verify that the message was sent
    return true;
  }

  return false;
}

void motr::WatchDir::execute_tasks() {
  for (const auto &task : tasks)
    if (!task.execute()) {
      MOTR_LOG("Task[{}] failed to execute", task.config.name);
      break;
    }
}
