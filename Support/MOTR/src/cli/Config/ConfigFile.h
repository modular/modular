//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_CONFIG_FILE_H
#define MOTR_CONFIG_FILE_H
#include "motr/Message.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace M::motr {
struct TaskConfig;
struct WatchConfig;
struct ServerConfig;
struct ConfigFile;

} // namespace M::motr

struct M::motr::TaskConfig {
  std::string name;
  std::string command;
  MessageType message = MessageType::None;
  bool broadcast = false;
};

struct M::motr::WatchConfig {
  std::string name;
  std::string path;
  std::vector<std::string> tasks;
  float latency;
};

struct M::motr::ServerConfig {
  std::string name;
  std::string documentRoot;
  std::string listeningPorts;
  int numThreads;
  int logLevel;
  std::string accessControlAllowOrigin;
};

struct M::motr::ConfigFile {
  static ConfigFile &getSingleton();
  static ConfigFile load(const std::string &path);

  static ConfigFile &initFromDisk();

  std::unordered_map<std::string, ServerConfig> servers;
  std::unordered_map<std::string, TaskConfig> tasks;
  std::unordered_map<std::string, WatchConfig> watches;

  const TaskConfig &getTask(const std::string &name) const;
  std::vector<TaskConfig> getTasks(const std::vector<std::string> &names) const;

  ConfigFile() = default;
};

#endif
