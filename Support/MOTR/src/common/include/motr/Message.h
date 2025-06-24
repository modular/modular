//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_MESSAGE_H
#define MOTR_MESSAGE_H

#ifdef MOTR_JSON_ENABLED
#include <nlohmann/json.hpp>
#endif

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#include "motr/Hash.h"

namespace M::motr {

// TODO: some of these are not used yet,
// so we should evaluate what to remove
enum class MessageType : uint8_t {
  None,
  Set,
  Span, // start and stop a timing span
  Stop,
  Status,
  Restart,
  Reload,     // reloads the gui app
  Thread,     // thread scope
  Process,    // process scope
  StackError, // when process exits without unwinding stack
  RPCCall,    // call a function
  RPCResult,  // result of an RPC call
  COUNT,
};

// determines layout of the message memory
enum class MessageFlags : uint8_t {
  None,
  Atomic,
  TagStr,
  TagInt,
  Push,
  Pop,
  COUNT,
};

static constexpr int MessageTypeCount = static_cast<int>(MessageType::COUNT);

struct Message {
  MessageType type;
  MessageFlags flags;
  uint8_t reserved[2];
  // todo: should procid be removed?
  uint32_t procid; // process id (redundant if pid can be resolved to root)
  uint64_t ts;     // timestamp, if flags == Tag* then contains value's hash
  uint64_t id;  // random unique id, if flags == Tag* then contains key's hash
  uint64_t pid; // parent id to this event (not process id!)

  bool isaTag(bool assertIfNot = false) const {
    bool result =
        flags == MessageFlags::TagStr || flags == MessageFlags::TagInt;
    if (assertIfNot)
      assert(result && "Invalid message type");
    return result;
  }

  uint64_t getTagValue() const {
    if (isaTag(true))
      return ts;
    return 0;
  }

  Hash::Value getTagValueHash() const { return Hash::Value(getTagValue()); }

  Hash::Value getTagKeyHash() const {
    if (isaTag(true))
      return Hash::Value(id);
    return 0;
  }

  std::string_view asStringView() const {
    return std::string_view(reinterpret_cast<const char *>(this),
                            sizeof(Message));
  }

  void copyFrom(std::string_view str) {
    assert(str.size() == sizeof(Message));
    std::copy(str.begin(), str.end(), reinterpret_cast<char *>(this));
  }

  bool operator==(const Message &other) const {
    return memcmp(this, &other, sizeof(Message)) == 0;
  }
};

static_assert(sizeof(Message) == 32, "Message size is not 32 bytes");

struct StringHeader {
  static constexpr const uint32_t Header = 0x4D305430; // M0T0
  StringHeader(const std::string_view &str)
      : header(Header), size(str.size()), hashId(M::motr::Hash::Value{str}.v) {}

  std::string_view asStringView() const {
    return std::string_view(reinterpret_cast<const char *>(this),
                            sizeof(StringHeader));
  }

  uint32_t header;
  uint32_t size;
  uint64_t hashId;
};

static_assert(sizeof(StringHeader) == 16, "StringHeader size is not 16 bytes");

inline constexpr const char *toString(const MessageType &type) {
  switch (type) {
  case MessageType::None:
    return "None";
  case MessageType::Set:
    return "Set";
  case MessageType::Span:
    return "Span";
  case MessageType::Stop:
    return "Stop";
  case MessageType::Status:
    return "Status";
  case MessageType::Restart:
    return "Restart";
  case MessageType::Reload:
    return "Reload";
  case MessageType::Thread:
    return "Thread";
  case MessageType::Process:
    return "Process";
  case MessageType::StackError:
    return "StackError";
  case MessageType::RPCCall:
    return "RPCCall";
  case MessageType::RPCResult:
    return "RPCResult";
  case MessageType::COUNT:
    return "COUNT";
  }
  return "Unknown";
}

inline constexpr const char *toString(const MessageFlags &flags) {
  switch (flags) {
  case MessageFlags::None:
    return "None";
  case MessageFlags::Atomic:
    return "Atomic";
  case MessageFlags::TagStr:
    return "TagStr";
  case MessageFlags::TagInt:
    return "TagInt";
  case MessageFlags::Push:
    return "Push";
  case MessageFlags::Pop:
    return "Pop";
  case MessageFlags::COUNT:
    return "COUNT";
  }
  return "Unknown";
}

template <typename T>
typename T::mapped_type findInOrNone(const std::string &str, const T &map) {
  auto it = map.find(str);
  if (it == map.end())
    return static_cast<typename T::mapped_type>(0);
  return it->second;
}

inline MessageType fromString(const std::string &str) {
  static std::unordered_map<std::string, MessageType> map = {
      {"None", MessageType::None},             //
      {"Set", MessageType::Set},               //
      {"Span", MessageType::Span},             //
      {"Stop", MessageType::Stop},             //
      {"Status", MessageType::Status},         //
      {"Restart", MessageType::Restart},       //
      {"Reload", MessageType::Reload},         //
      {"Thread", MessageType::Thread},         //
      {"Process", MessageType::Process},       //
      {"StackError", MessageType::StackError}, //
      {"RPCCall", MessageType::RPCCall},       //
      {"COUNT", MessageType::COUNT},           //
  };
  return findInOrNone(str, map);
}

inline MessageFlags messageFlagsFromString(const std::string &str) {
  static std::unordered_map<std::string, MessageFlags> map = {
      {"Atomic", MessageFlags::Atomic}, //
      {"TagStr", MessageFlags::TagStr}, //
      {"TagInt", MessageFlags::TagInt}, //
      {"Push", MessageFlags::Push},     //
      {"Pop", MessageFlags::Pop},       //
  };
  return findInOrNone(str, map);
}

#ifdef MOTR_JSON_ENABLED
inline std::string toJSONString(const M::motr::Message &message) {
  nlohmann::json json;
  json["type"] = toString(message.type);
  json["flags"] = toString(message.flags);
  json["procid"] = message.procid;
  json["ts"] = message.ts;
  json["id"] = message.id;
  json["pid"] = message.pid;
  return json.dump();
}

inline Message fromJSONStringView(std::string_view jsonString) {
  nlohmann::json json = nlohmann::json::parse(jsonString);
  Message message;
  message.type = fromString(json["type"]);
  message.flags = messageFlagsFromString(json["flags"]);
  message.procid = json["procid"];
  message.ts = json["ts"];
  message.id = json["id"];
  message.pid = json["pid"];
  return message;
}

inline Message fromJSONString(const std::string &jsonString) {
  return fromJSONStringView(std::string_view(jsonString));
}

#endif

} // namespace M::motr
#endif
