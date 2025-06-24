//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#define MOTR_JSON_ENABLED 1
#include "Config/ConfigFile.h"

#include "motr/Common.h"
#include "motr/Log.h"
#include "motr/motr.h"

#include <functional>
#include <string>
#include <unordered_map>

#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

// Forward declarations of subcommand's main functions.
int serverMain(int argc, char **argv);
int testMain(int argc, char **argv);
int statusMain(int argc, char **argv);
int cleanMain(int argc, char **argv);
int flagsMain(int argc, char **argv);
int stringsMain(int argc, char **argv);
int tagsMain(int argc, char **argv);
std::unordered_map<std::string, std::function<int(int, char **)>> command_map =
    {
        // clang-format off
        {"test", testMain},
        {"status", statusMain},
        {"clean", cleanMain},
        {"server", serverMain},
        {"flags", flagsMain},
        {"strings", stringsMain},
        {"tags", tagsMain},
        // clang-format on
};

int main(int argc, char **argv) {
  using namespace M::motr;
  auto startTimestamp = Time::getStartTimestamp().v;

  if (argc > 1 && strcmp(argv[1], "server") == 0) {
    // Server mode always cleans up any possibly leftover SharedMemory
    // mailboxes./
    cleanMain(0, nullptr);
    // After cleaning, checking valid() will recreate new mailboxes.
    ServerInbox::valid();
    ServerInboxString::valid();
  }

  MOTR_TraceProgram(mainTrace, "motr");
  MOTR_TagStr("git_commit", GIT_COMMIT);
  MOTR_TagStr("version", MOTR_VERSION_STRING);
  MOTR_TagIntVar(return_value, "return_value", 0);

  MOTR_LOG("motr version {} ({})", MOTR_VERSION_STRING, GIT_COMMIT);
  auto buildTimestamp = Time::getBuildTimestamp();
  Time::Duration elapsedSinceBuild = startTimestamp - buildTimestamp;
  MOTR_LOG("build_time={} ({} ago)", buildTimestamp.toString(),
           elapsedSinceBuild.toString());

  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    MOTR_Trace(version);
    return_value = 0;
    return return_value;
  }

  [[maybe_unused]] auto &config = M::motr::ConfigFile::initFromDisk();

  if (argc < 2) {
    MOTR_Trace(usage);
    MOTR_LOG("Usage: {} <command>", argv[0]);
    MOTR_LOG("{}", "Commands:");
    for (const auto &[name, func] : command_map) {
      MOTR_LOG("  {}", name);
    }
    return_value = 1;
    return return_value;
  }

  auto command = command_map.find(argv[1]);
  if (command == command_map.end()) {
    MOTR_Trace(unknownCommand);
    MOTR_LOG("Unknown command: {}", argv[1]);
    return_value = 1;
    return return_value;
  }

  return_value = command->second(argc, argv);
  return return_value;
}
