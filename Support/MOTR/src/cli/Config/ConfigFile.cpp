//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConfigFile.h"
#include "motr/Log.h"
#include "motr/Message.h"
#include <fstream>
#include <iostream>

// https://github.com/biojppm/rapidyaml/blob/d8f4d0150b3f84dca5d2775154f224e0c580abc5/README.md#single-header-file
#define RYML_SINGLE_HDR_DEFINE_NOW
#include "rapidyaml-0.7.2.hpp"

using namespace M;

motr::ConfigFile &motr::ConfigFile::getSingleton() {
  static motr::ConfigFile singleton;
  return singleton;
}

static ryml::Tree parse_yaml(const std::string &path) {
  std::ifstream file(path);
  std::string content((std::istreambuf_iterator<char>(file)), {});
  ryml::Tree tree = ryml::parse_in_arena(c4::csubstr(content.c_str()));
  // TODO: handle parse errors and show location of error
  return tree;
}

static bool parseTasks(ryml::NodeRef tasks,
                       decltype(motr::ConfigFile::tasks) &tasks_map) {
  if (!tasks.readable())
    return false;

  for (ryml::ConstNodeRef task : tasks.children()) {
    const auto &key = task.key();
    std::string name(key.str, key.len);
    motr::TaskConfig &tc = tasks_map[name];
    tc.name = name;

    task.get_if<std::string>("command", &tc.command);
    task.get_if<bool>("broadcast", &tc.broadcast);

    std::string tmp_message;
    if (task.get_if<std::string>("message", &tmp_message))
      tc.message = motr::fromString(tmp_message);
  }
  return true;
}

std::string keyString(const c4::yml::NodeRef &node) {
  const auto &key = node.key();
  return std::string(key.str, key.len);
}

std::string valString(const c4::yml::NodeRef &node) {
  const auto &val = node.val();
  return std::string(val.str, val.len);
}

motr::ServerConfig parseServer(ryml::NodeRef node) {
  motr::ServerConfig sc;
  sc.name = keyString(node);

  node.get_if<std::string>("document_root", &sc.documentRoot);
  node.get_if<std::string>("listening_ports", &sc.listeningPorts);
  node.get_if<int>("num_threads", &sc.numThreads);
  node.get_if<int>("log_level", &sc.logLevel);
  node.get_if<std::string>("access_control_allow_origin",
                           &sc.accessControlAllowOrigin);

  return sc;
}

static bool parseServers(ryml::NodeRef servers,
                         decltype(motr::ConfigFile::servers) &servers_map) {
  if (!servers.readable())
    return false;

  for (ryml::NodeRef server : servers.children())
    servers_map[keyString(server)] = parseServer(server);

  return true;
}

static motr::WatchConfig parseWatch(ryml::NodeRef node) {
  motr::WatchConfig wc;
  wc.name = keyString(node);
  node.get_if<std::string>("path", &wc.path);
  node.get_if<float>("latency", &wc.latency);
  auto tasks = node["tasks"];
  if (tasks.is_seq()) {
    for (ryml::NodeRef task : tasks.children()) {
      wc.tasks.push_back(valString(task));
    }
  }
  return wc;
}

static bool parseWatches(ryml::NodeRef node,
                         decltype(motr::ConfigFile::watches) &watches_map) {
  if (!node.readable())
    return false;

  for (ryml::NodeRef watch : node.children())
    watches_map[keyString(watch)] = parseWatch(watch);
  return true;
}

motr::ConfigFile motr::ConfigFile::load(const std::string &path) {
  motr::ConfigFile config;
  auto tree = parse_yaml(path);

  if (tree.rootref().readable()) {
    parseTasks(tree.rootref()["tasks"], config.tasks);
    parseServers(tree.rootref()["servers"], config.servers);
    parseWatches(tree.rootref()["watch"], config.watches);
  }

  return config;
}

const motr::TaskConfig &
motr::ConfigFile::getTask(const std::string &name) const {
  return tasks.at(name);
}

std::vector<motr::TaskConfig>
motr::ConfigFile::getTasks(const std::vector<std::string> &names) const {
  std::vector<TaskConfig> result;
  result.reserve(names.size());
  for (const auto &name : names)
    result.push_back(tasks.at(name));
  return result;
}

motr::ConfigFile &motr::ConfigFile::initFromDisk() {
  auto &config = getSingleton();
  // todo: check ancestor paths and environemnt variable
  config = load("config.yaml");

  return config;
}
