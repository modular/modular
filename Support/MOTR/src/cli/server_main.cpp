//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#define MOTR_JSON_ENABLED 1

#include "Config/ConfigFile.h"
#include "HostInfo.h"
#include "Watch/WatchDir.h"
#include "Webserver/WebServer.h"
#include "motr/Common.h"
#include "motr/EventTree.h"
#include "motr/RPCMethods.h"
#include "motr/motr.h"

#if defined(MOTR_PLATFORM_MACOS)
#include <CoreFoundation/CoreFoundation.h>
#endif
#include <list>
#include <unistd.h>

using namespace M::motr;

int cleanMain(int argc, char **argv);
int statusMain(int argc, char **argv);

using Servers = std::vector<std::unique_ptr<WebServer>>;

namespace M::motr::RPC::Methods {
getHostInfo::FuncType getHostInfo::func = M::motr::getHostInfo;
getMotrServerInfo::FuncType getMotrServerInfo::func =
    M::motr::getMotrServerInfo;
} // namespace M::motr::RPC::Methods

uint64_t getRequestId(const EventTreeNode::Ptr &node) {
  std::optional<uint64_t> requestId =
      node->getChildTagValue<uint64_t>({Constants::__rpc_request_id__::hash});
  if (!requestId) {
    MOTR_LOG("{}", "Request ID not found");
    return 0;
  }
  return *requestId;
}

template <typename T, typename... Args>
bool executeRPCMethodAndSendRPCResult(EventTreeNode::Ptr &node,
                                      Args &&...args) {
  auto result = RPC::Methods::execute<T>(std::forward<Args>(args)...);
  if (!result) {
    MOTR_LOG("{}", "RPC Method execution failed");
    return false;
  }
  RPC::sendRPCResult(*result, getRequestId(node));
  node->setParent(nullptr);
  return true;
}

void processEventTree() {
  static Time::Timestamp lastTime;
  Time::Timestamp now = Time::Timestamp::now();
  if (now - lastTime < Time::Duration::fromSeconds(5)) {
    return;
  }
  lastTime = now;

  std::vector<EventTreeNode::Ptr> rpcCallNodes;

  EventTree &eventTree = EventTree::getSingleton();
  for (auto &node : eventTree.getAllNodes()) {
    std::string json = toJSONString(node->message);
    size_t depth = node->numAncestors();
    std::string indent(depth, '-');
    MOTR_LOG("{}> {}", indent, json);
    if (node->message.type == MessageType::RPCCall &&
        node->message.flags == MessageFlags::Push) {
      rpcCallNodes.push_back(node);
    }
  }

  for (auto &node : rpcCallNodes) {
    auto rpcCallName =
        node->getChildTagValue<MString>({Constants::__rpc_call_name__::hash});
    if (!rpcCallName)
      MOTR_LOG("{}", "RPC Call Name not found");
    else {
      MOTR_LOG("RPC Call: {}", rpcCallName->sv());
      switch (rpcCallName->hash.v) {
      case RPC::Methods::getHostInfo::rpcCallNameHash.v:
        executeRPCMethodAndSendRPCResult<RPC::Methods::getHostInfo>(node);
        break;
      case RPC::Methods::getMotrServerInfo::rpcCallNameHash.v:
        executeRPCMethodAndSendRPCResult<RPC::Methods::getMotrServerInfo>(node);
        break;
      default:
        MOTR_LOG("Unhandled RPC Call: {}", rpcCallName->sv());
        break;
      }
    }
  }
}

void writeChar(char c) {
  [[maybe_unused]] auto unused0000 = write(1, &c, 1);
  [[maybe_unused]] auto unused0001 = fsync(1);
}

struct ProgressDots {
  int dots = 0;

  ProgressDots() = default;

  static ProgressDots &instance() {
    static ProgressDots singleton;
    return singleton;
  }

  std::string optnewline() {
    if (!dots)
      return "";
    dots = 0;
    return "\n";
  };

  void print() {
    // Periodic dot to indicate progress
    writeChar('.');

    if (++dots % 40 == 0) {
      writeChar('\n');
      dots = 0;
    }
  }
};

static size_t logMessages(const std::vector<Message> &messages,
                          const std::string &serverElapsedStr) {
  const std::string emptyIsoDate(27, ' ');
  size_t N = messages.size();
  int64_t now = Time::Timestamp::now().nanoseconds();
  size_t sendCount = 0;

  for (size_t n = 0; n < N; n++) {
    auto &msg = messages[n];
    sendCount++;
    const bool isTagStr = msg.flags == MessageFlags::TagStr;
    const bool isTagInt = msg.flags == MessageFlags::TagInt;
    const bool isTag = isTagStr || isTagInt;

    std::string iso_date = isTag ? emptyIsoDate : timeNsToISODate(msg.ts);
    std::string slug = isTag ? "TAG" : "MSG";
    std::string info;

    if (isTagStr) {
      M::motr::MString key{msg.id, false};
      M::motr::MString value{msg.getTagValue(), false};
      auto valuesv = value.sv(true);
      size_t len = valuesv.size();
      std::string summary = summaryString(valuesv, 30);
      info = fmt::format("[\"{}\"][{}] = \"{}\"", key.str(true), len, summary);
    } else if (isTagInt) {
      M::motr::MString key{msg.id, false};
      uint64_t value = msg.getTagValue();
      info =
          fmt::format("[\"{}\"] = 0x{:016x} ({})", key.str(true), value, value);
    } else {
      info = fmt::format("type={} flags={}", toString(msg.type),
                         toString(msg.flags));
    }

    MOTR_LOG("{}[{}] {} [{:2d}/{:2d}] [{:>7s}] 0x{:016x} ^0x{:016x} {}",
             ProgressDots::instance().optnewline(), iso_date, slug, sendCount,
             N, serverElapsedStr, msg.id, msg.pid, info);
  }

  return sendCount;
}
void processStringMessages(ServerInboxString &stringInbox, Servers &servers,
                           const std::string &serverElapsedStr) {
  M::motr::StringQueueResult stringMessages =
      stringInbox.getStringQueue().recv();
  size_t N = stringMessages.headers.size();
  std::unordered_map<uint64_t, std::string_view> stringMap;
  for (size_t i = 0; i < N; ++i) {
    stringMap[stringMessages.headers[i]->hashId] = stringMessages.views[i];
  }
  size_t n = 0;
  for (const auto &[hashId, str] : stringMap) {
    ++n;
    auto nowstr = Time::Timestamp::now().toString();
    MOTR_LOG("{}[{}] STR [{:2d}/{:2d}] [{:>7s}] 0x{:016x} [{:3d}] \"{}\"",
             ProgressDots::instance().optnewline(), nowstr, n, N,
             serverElapsedStr, hashId, str.size(), summaryString(str, 50));
  }

  for (const auto &[hashId, str] : stringMap)
    for (auto &server : servers)
      server->sendWebsocketText(str);
}

int processMessageQueue(ServerInbox &inbox, Servers &servers,
                        const std::string &serverElapsedStr) {
  std::vector<Message> messages;
  messages = inbox.recv(1024);
  if (messages.empty())
    return 0;

  logMessages(messages, serverElapsedStr);

  for (auto &message : messages) {
    if (message.type == MessageType::Stop) {
      MOTR_LOG("STOP...", "");
      return -1;
    }
  }

  // Send the batch of messages as a single binary message to all
  // websocket clients
  size_t nbytes = messages.size() * sizeof(Message);
  if (nbytes > 0) {
    MOTR_LOG("Sending {} bytes to {} servers", nbytes, servers.size());
    std::string_view dataSV{reinterpret_cast<const char *>(messages.data()),
                            nbytes};
    for (auto &server : servers)
      server->sendWebsocketBinary(dataSV);
  }

  return messages.size();
}

// Periodic ping messages to keep connection alive
void sendPeriodicStatusMessage(const Time::Timestamp &now,
                               uint64_t serverPingIntervalMs) {
  static Time::Timestamp lastPingTime;

  const Time::Duration interval =
      Time::Duration::fromMilliseconds(serverPingIntervalMs);
  if (now - lastPingTime > interval) {
    lastPingTime = now;
    EmitMessage<MessageType::Status, MessageFlags::Atomic>{};
  }
}

int serverMain(int argc, char **argv) {
  Time::Timestamp startTimestamp = Time::Timestamp::now();
  /*
  {
    MOTR_FlagInt(serverDontStealMailbox);
    if (!serverDontStealMailbox) {
      // MOTR_Trace(serverResetMailboxes);
      cleanMain(0, nullptr);
      M::motr::ServerInbox::valid();
      M::motr::ServerInboxString::valid();
    }
  }
  */

  ServerInbox inbox;

  if (!inbox.valid()) {
    MOTR_Trace(invalidInbox);
    MOTR_LOG("Inbox is not valid\n", "");
    return 1;
  }

  ServerOutbox outbox;
  MOTR_Trace(server);

  // call statusMain to print debug info to the console
  // on server startup
  statusMain(0, nullptr);

  if (!outbox.valid()) {
    MOTR_Trace(invalidOutbox);
    MOTR_LOG("Outbox is not valid\n", "");
    // Outbox::getQueuePtr().reset();
  }

  ServerInboxString stringInbox;
  if (!stringInbox.valid()) {
    MOTR_Trace(invalidStringInbox);
    MOTR_LOG("String inbox is not valid\n", "");
    return 1;
  }

  MOTR_LOG("Starting server...\n", "");
  statusMain(0, nullptr);

  auto &config = ConfigFile::getSingleton();

  std::list<WatchDir> watches;
  {
    MOTR_Trace(configureWatches);
    for (const auto &[name, watch_config] : config.watches)
      watches.emplace_back(watch_config);

    // start all watches after all are instantiated
    for (auto &watch : watches)
      watch.start();
  }

  Servers servers;
  for (const auto &[name, server_config] : config.servers)
    servers.push_back(std::make_unique<WebServer>(server_config));

  MOTR_FlagInt(serverQuit);
  MOTR_FlagInt(serverCycleTimeMs);
  MOTR_FlagInt(serverPingIntervalMs);
  serverCycleTimeMs = 100;
  serverPingIntervalMs = 10000;
  serverQuit = 0;

  Time::Elapsed serverElapsed;
  while (!serverQuit) {
    Time::Timestamp now = Time::Timestamp::now();

    // nextCycleStart is stored as a Duration
    // because it is measured from start of server (not epoch)
    Time::Duration nextCycleStart;
    {
      // convert sync flag from ms to ns
      Time::Duration serverCycleTimeNs =
          Time::Duration::fromMilliseconds(serverCycleTimeMs);
      int64_t period_ns = serverCycleTimeNs.nanoseconds();
      int64_t serverMark = serverElapsed.elapsed().nanoseconds();
      nextCycleStart.v = (serverMark / period_ns) * period_ns + period_ns;
      assert(nextCycleStart > serverMark);
    }

    const std::string serverElapsedStr = serverElapsed.toString();

    sendPeriodicStatusMessage(now, serverPingIntervalMs);

    if (!watches.empty()) {
      for (auto &watch : watches)
        watch.checkPendingExecution();
#if defined(MOTR_PLATFORM_MACOS)
      CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, true);
#endif
    }

    int nmsg = 0;
    do {
      // processEventTree();

      processStringMessages(stringInbox, servers, serverElapsedStr);

      nmsg = processMessageQueue(inbox, servers, serverElapsedStr);
      if (nmsg == -1) {
        serverQuit = 1;
      }

    } while (nmsg > 0);

    if (serverQuit)
      break;

    auto durationUntilNextCycleStart = nextCycleStart - serverElapsed.elapsed();
    int64_t sleep_ns = durationUntilNextCycleStart.nanoseconds();
    if (sleep_ns > 0) {
      ProgressDots::instance().print();
      std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns));
    }
  }
  return 0;
}
