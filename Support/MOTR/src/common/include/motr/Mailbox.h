//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_MAILBOX_H
#define MOTR_MAILBOX_H

#include "motr/Log.h"
#include "motr/Message.h"
#include "motr/Namespace.h"
#include "motr/Queue.h"
#include "motr/SharedMemory.h"

#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace M::motr {

enum class MailboxRealm {
  GlobalEvents,
  GlobalStrings,
  ProcessEvents,
  ProcessStrings,
};

enum class MailboxScope {
  Global,
  Process,
};

enum class MailboxMessageType {
  Message,
  StringMessage,
};

enum class MailboxDirection {
  Inbox, // can send and receive
  Outbox,
};

template <MailboxRealm mailbox_realm>
struct MailboxRealms {};

template <MailboxScope mailbox_scope>
struct MailboxScopes {};

template <MailboxMessageType mailbox_message_type>
struct MailboxMessageTypes {};

template <MailboxDirection mailbox_owner>
struct MailboxDirections {};

template <>
struct MailboxScopes<MailboxScope::Global> {
  static constexpr MailboxScope mailboxScope = MailboxScope::Global;
  static constexpr bool isGlobal = true;
  static uint32_t getProcessId() { return 0; }
};

template <>
struct MailboxScopes<MailboxScope::Process> {
  static constexpr MailboxScope mailboxScope = MailboxScope::Process;
  static constexpr bool isGlobal = false;
  static uint32_t getProcessId() { return (uint32_t)getpid(); }
};

template <>
struct MailboxDirections<MailboxDirection::Inbox> {
  static constexpr bool isInbox = true;
  static constexpr SharedMemoryInit shmMode = SharedMemoryInit::ExclusiveCreate;
};

template <>
struct MailboxDirections<MailboxDirection::Outbox> {
  static constexpr bool isInbox = false;
  static constexpr SharedMemoryInit shmMode = SharedMemoryInit::OpenExisting;
};

template <>
struct MailboxMessageTypes<MailboxMessageType::Message> {
  using MessageType = Message;
  using MessageQueue = Queue<MessageType>;
  using MessageQueuePtr = std::unique_ptr<MessageQueue>;
  static constexpr MailboxMessageType mailboxMessageType =
      MailboxMessageType::Message;
  static constexpr size_t messageSize = sizeof(MessageType);
  static constexpr bool isFixedSize = true;
  static constexpr const char *messageTypeName = "msg";
  static constexpr bool needsStringHeader = false;
};

template <>
struct MailboxMessageTypes<MailboxMessageType::StringMessage> {
  using MessageType = char;
  using MessageQueue = Queue<MessageType>;
  using MessageQueuePtr = std::unique_ptr<MessageQueue>;
  static constexpr MailboxMessageType mailboxMessageType =
      MailboxMessageType::StringMessage;
  static constexpr bool isFixedSize = false;
  static constexpr const char *messageTypeName = "str";
  static constexpr bool needsStringHeader = true;
};

template <>
struct MailboxRealms<MailboxRealm::GlobalEvents>
    : MailboxMessageTypes<MailboxMessageType::Message>,
      MailboxScopes<MailboxScope::Global> {
  static constexpr MailboxRealm mailboxRealm = MailboxRealm::GlobalEvents;
  static constexpr size_t capacity = 1024 * 1024; // in number of messages
  static constexpr size_t totalMemorySize = capacity * messageSize;
};

static_assert(MailboxRealms<MailboxRealm::GlobalEvents>::totalMemorySize ==
                  32 * 1024 * 1024,
              "Global events mailbox should be 32MB");

template <>
struct MailboxRealms<MailboxRealm::GlobalStrings>
    : MailboxMessageTypes<MailboxMessageType::StringMessage>,
      MailboxScopes<MailboxScope::Global> {
  static constexpr MailboxRealm mailboxRealm = MailboxRealm::GlobalStrings;
  static constexpr size_t mb = 32;
  static constexpr size_t capacity = mb * 1024 * 1024;
  static constexpr size_t totalMemorySize = mb * 1024 * 1024;
};

static_assert(MailboxRealms<MailboxRealm::GlobalStrings>::totalMemorySize ==
                  32 * 1024 * 1024,
              "Global strings mailbox should be 32MB");

template <>
struct MailboxRealms<MailboxRealm::ProcessEvents>
    : MailboxMessageTypes<MailboxMessageType::Message>,
      MailboxScopes<MailboxScope::Process> {
  static constexpr MailboxRealm mailboxRealm = MailboxRealm::ProcessEvents;
  static constexpr size_t capacity = 1024; // in number of messages
  static constexpr size_t totalMemorySize = capacity * messageSize;
};

static_assert(MailboxRealms<MailboxRealm::ProcessEvents>::totalMemorySize ==
                  32 * 1024,
              "Process events mailbox should be 32KB");

template <>
struct MailboxRealms<MailboxRealm::ProcessStrings>
    : MailboxMessageTypes<MailboxMessageType::StringMessage>,
      MailboxScopes<MailboxScope::Process> {
  static constexpr MailboxRealm mailboxRealm = MailboxRealm::ProcessStrings;
  static constexpr size_t totalMemorySize = 1024 * 32;
};

static_assert(MailboxRealms<MailboxRealm::ProcessStrings>::totalMemorySize ==
                  32 * 1024,
              "Process strings mailbox should be 32KB");

template <MailboxRealm mailbox_realm, MailboxDirection mailbox_owner>
struct MailboxConfigs : MailboxRealms<mailbox_realm>,
                        MailboxDirections<mailbox_owner> {
  using Realm = MailboxRealms<mailbox_realm>;
  using Direction = MailboxDirections<mailbox_owner>;
  using StringQueueType = StringQueue;

  static std::string name() {
    static std::string str;
    static int generation = 0;

    if (generation < Namespace::generation()) {
      generation = Namespace::generation();
      str =
          Namespace::makeSHMName(Realm::getProcessId(), Realm::messageTypeName);
    }

    return str;
  }
};

using GlobalEventsInboxConfig =
    MailboxConfigs<MailboxRealm::GlobalEvents, MailboxDirection::Inbox>;
using GlobalEventsOutboxConfig =
    MailboxConfigs<MailboxRealm::GlobalEvents, MailboxDirection::Outbox>;

using GlobalStringsInboxConfig =
    MailboxConfigs<MailboxRealm::GlobalStrings, MailboxDirection::Inbox>;
using GlobalStringsOutboxConfig =
    MailboxConfigs<MailboxRealm::GlobalStrings, MailboxDirection::Outbox>;

/*
using ProcessEventsInboxConfig =
    MailboxConfigs<MailboxRealm::ProcessEvents, MailboxDirection::Inbox>;
using ProcessEventsOutboxConfig =
    MailboxConfigs<MailboxRealm::ProcessEvents, MailboxDirection::Outbox>;

using ProcessStringsInboxConfig =
    MailboxConfigs<MailboxRealm::ProcessStrings, MailboxDirection::Inbox>;
using ProcessStringsOutboxConfig =
    MailboxConfigs<MailboxRealm::ProcessStrings, MailboxDirection::Outbox>;

  */

struct SendMessagesCallbackQueue {
  using SendMessagesCallback =
      std::function<size_t(const Message *first, size_t count)>;
  SendMessagesCallback sendMessagesCallback;

  SendMessagesCallbackQueue(SharedMemoryInit, const std::string &, size_t) {}

  bool valid() const { return sendMessagesCallback != nullptr; }

  size_t send(const Message *first, size_t count) {
    assert(sendMessagesCallback);
    if (sendMessagesCallback) {
      return sendMessagesCallback(first, count);
    }
    return 0;
  }
};

struct SendStringViewsCallbackQueue {
  using StringViews = std::vector<std::string_view>;
  using SendStringViewsCallback =
      std::function<size_t(const StringViews &strs)>;
  SendStringViewsCallback sendStringViewsCallback;

  SendStringViewsCallbackQueue(SharedMemoryInit, const std::string &, size_t) {}
  SendStringViewsCallbackQueue(const SendStringViewsCallbackQueue &) = default;

  bool valid() const { return sendStringViewsCallback != nullptr; }

  size_t send(const StringViews &strs) {
    assert(sendStringViewsCallback);
    if (sendStringViewsCallback) {
      return sendStringViewsCallback(strs);
    }
    return 0;
  }
};

struct SendMessagesCallbackMailboxConfig {
  using MessageQueue = SendMessagesCallbackQueue;
  using MessageQueuePtr = std::unique_ptr<SendMessagesCallbackQueue>;
  using MessageType = Message;
  static constexpr SharedMemoryInit shmMode = SharedMemoryInit::ExclusiveCreate;
  static std::string name() { return "send_messages_callback_mailbox"; }
  static constexpr size_t capacity = 1024;
  static constexpr bool isInbox = true;
  static constexpr bool isOutbox = false;
  static constexpr bool isString = false;
  static constexpr bool isFixedSize = true;
};

struct SendStringViewsCallbackMailboxConfig {
  using MessageQueue = SendStringViewsCallbackQueue;
  using MessageQueuePtr = std::unique_ptr<SendStringViewsCallbackQueue>;
  using MessageType = char;
  using StringQueueType = SendStringViewsCallbackQueue;
  static constexpr SharedMemoryInit shmMode = SharedMemoryInit::ExclusiveCreate;
  static constexpr MailboxMessageType mailboxMessageType =
      MailboxMessageType::StringMessage;
  static std::string name() { return "send_string_views_callback_mailbox"; }
  static constexpr size_t capacity = 1024;
  static constexpr bool isInbox = false;
  static constexpr bool isOutbox = true;
  static constexpr bool isString = true;
  static constexpr bool isFixedSize = false;
  static constexpr bool needsStringHeader = false;
};

template <typename MailboxConfig>
struct MailboxBase {
  using Config = MailboxConfig;
  using Queue = typename MailboxConfig::MessageQueue;
  using QueuePtr = typename MailboxConfig::MessageQueuePtr;
  using Message = typename MailboxConfig::MessageType;

  static bool valid() {
    auto &queue = getQueuePtr();
    return queue && queue->valid();
  }

  static QueuePtr &getQueuePtrRef() {
    static QueuePtr queue_ptr;
    return queue_ptr;
  }

  static QueuePtr &getQueuePtr() {
    QueuePtr &queue_ptr = getQueuePtrRef();

    // todo: remove initialization on demand and require
    // explicit lifetime management
    // this would make send/recv a bit faster as it doesn't need to check if the
    // queue is initialized for every message sent/received
    if (false) {
      static std::once_flag once;
      std::call_once(once, [&]() {
        queue_ptr = std::make_unique<Queue>(MailboxConfig::shmMode,
                                            MailboxConfig::name().c_str(),
                                            MailboxConfig::capacity);
      });
    } else {
      if (!queue_ptr) {
        queue_ptr = std::make_unique<Queue>(MailboxConfig::shmMode,
                                            MailboxConfig::name().c_str(),
                                            MailboxConfig::capacity);
      }
    }
    assert(queue_ptr != nullptr);
    return queue_ptr;
  }

  static Queue &getQueue() { return *getQueuePtr(); }

  // send multiple messages
  static size_t send(const Message *first, size_t count) {
    return getQueue().send(first, count);
  }

  // send one message
  static size_t send(const Message &msg) { return send(&msg, 1); }

  // receive multiple messages
  // maxcount is the maximum number of messages to receive
  // if maxcount is 0, all messages in the queue will be received
  template <typename T = MailboxConfig>
  static std::vector<Message> recv(size_t maxcount = 1,
                                   std::enable_if_t<T::isInbox, int> = 0) {
    return getQueue().recv(maxcount);
  }
};

template <typename MailboxConfig, typename = void>
struct Mailbox : MailboxBase<MailboxConfig> {
  using MailboxBase<MailboxConfig>::recv;
  using MailboxBase<MailboxConfig>::send;
};

// Specialization for StringMessage
template <typename MailboxConfig>
struct Mailbox<MailboxConfig,
               std::enable_if_t<MailboxConfig::mailboxMessageType ==
                                MailboxMessageType::StringMessage>>
    : MailboxBase<MailboxConfig> {

  using StringViews = typename MailboxConfig::StringQueueType::StringViews;
  using StringQueueType = typename MailboxConfig::StringQueueType;

  static StringQueueType &getStringQueue() {
    static StringQueueType queue(MailboxBase<MailboxConfig>::getQueue());
    return queue;
  }

  static bool valid() { return getStringQueue().valid(); }

  static size_t send(const std::vector<std::string_view> &strs) {
    auto &queue = getStringQueue();
    if (!queue.valid()) {
      MOTR_LOG("Mailbox send(size={}) invalid queue", strs.size());
      return 0;
    }

    std::vector<StringHeader> headers;
    StringViews allViews;

    // todo: use a bloom filter for each client to avoid duplicates
    static std::unordered_set<Hash::Value> cache;
    constexpr bool useCache = false;

    if constexpr (MailboxConfig::needsStringHeader)
      headers.reserve(strs.size());

    allViews.reserve(strs.size() * 2);
    for (const auto &str : strs) {
      if (str.empty())
        continue;
      Hash::Value hash{str};

      if constexpr (useCache) {
        if (cache.find(hash) != cache.end())
          continue;
        cache.insert(hash);
      }

      if constexpr (MailboxConfig::needsStringHeader) {
        headers.emplace_back(str);
        allViews.emplace_back(headers.back().asStringView());
      }

      allViews.emplace_back(str);
    }
    // MOTR_LOG("Mailbox send(size={})", allViews.size());
    // getStringQueue().debugPrint();
    return getStringQueue().send(allViews);
  }

  template <typename T = MailboxConfig>
  std::vector<std::pair<uint64_t, std::string>>
  recv(std::enable_if_t<T::isInbox, int> = 0) {
    StringQueueResult results = getStringQueue().recv();
    std::vector<std::pair<uint64_t, std::string>> map;
    for (size_t i = 0; i < results.headers.size(); ++i) {
      const auto &header = results.headers[i];
      const auto &view = results.views[i];
      map.emplace_back(header->hashId, std::string(view));
    }
    return map;
  }
};

#ifdef __EMSCRIPTEN__

using ServerInbox = Mailbox<SendMessagesCallbackMailboxConfig>;
using ServerOutbox = Mailbox<SendMessagesCallbackMailboxConfig>;
using ServerInboxString = Mailbox<SendStringViewsCallbackMailboxConfig>;
using ServerOutboxString = Mailbox<SendStringViewsCallbackMailboxConfig>;

#else

using ServerInbox = Mailbox<GlobalEventsInboxConfig>;
using ServerOutbox = Mailbox<GlobalEventsOutboxConfig>;
using ServerInboxString = Mailbox<GlobalStringsInboxConfig>;
using ServerOutboxString = Mailbox<GlobalStringsOutboxConfig>;

#endif

} // namespace M::motr

#endif // MOTR_MAILBOX_H
