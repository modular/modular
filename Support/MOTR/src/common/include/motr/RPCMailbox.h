//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_RPCMAILBOX_H
#define MOTR_RPCMAILBOX_H

#include "motr/EventTree.h"
#include "motr/Hash.h"
#include "motr/Log.h"
#include "motr/MString.h"
#include "motr/Mailbox.h"
#include "motr/Message.h"
#include "motr/RPC.h"

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace M::motr::RPC {

// Compute the fingerprint hash of a EventTree and tag children
// to validate the RPCResult has all of the expected children and no extra ones.
inline uint64_t getFingerprint(const EventTreeNode::Ptr &node);

// Serialize a RPC IsReflectable type to a series of Mailbox messages
template <typename T, typename = std::enable_if_t<is_rpc_reflectable<T>::value>>
void sendRPCResult(const T &value, const uint64_t requestId);

// Deserialize a RPC IsReflectable type from an EventTreeNode tree
// that was constructed from a series of Mailbox messages.
template <typename T, typename = std::enable_if_t<is_rpc_reflectable<T>::value>>
std::optional<T> getRPCResult(const EventTreeNode::Ptr &node);

template <const char *NAME, typename RESULT_TYPE>
struct RPCCall {
  using ResultType = std::optional<RESULT_TYPE>;
  static constexpr std::string_view name{NAME};
  static constexpr Hash::Value nameHash{NAME};
  static constexpr uint64_t ResultFingerprint = RESULT_TYPE::RPCFingerprint;
  Hash::Value requestId;

  ResultType wait(size_t timeout_ms) { return ResultType{}; }

  operator ResultType() { return wait(); }

  RPCCall() {
    requestId = Hash::Value{getNextMessageID()};
    std::vector<std::string_view> strings = {
        name, Constants::__rpc_call_name__::sv,
        Constants::__rpc_fingerprint__::sv, Constants::__rpc_request_id__::sv};
    ServerOutboxString::send(strings);
    Span<MessageType::RPCCall>
        rpcCallSpan{}; // todo: determind if parent should be 0
    TagStr(Constants::__rpc_call_name__::hash, nameHash);
    TagInt(Constants::__rpc_fingerprint__::hash, ResultFingerprint);
    TagInt(Constants::__rpc_request_id__::hash, requestId.v);
  }
};

} // namespace M::motr::RPC

inline uint64_t M::motr::RPC::getFingerprint(const EventTreeNode::Ptr &node) {
  Hash::SetFingerprint fingerprint;
  for (auto &child : node->children) {
    const Message &msg = child->message;
    if (msg.isaTag()) {
      fingerprint.add(msg.id);
    }
  }
  return fingerprint.get();
}

template <typename T, typename>
inline void M::motr::RPC::sendRPCResult(const T &value,
                                        const uint64_t requestId) {
  // MOTR_LOG("WebServer[{}] sendRPCResult", server->config.name);
  std::vector<std::string_view> strings;
  Span<MessageType::RPCResult> span;
  MString fingerprint{Constants::__rpc_fingerprint__::sv};
  MString requestIdKey{Constants::__rpc_request_id__::sv};
  motr::TagInt(fingerprint.hash, value.RPCFingerprint);
  motr::TagInt(requestIdKey.hash, requestId);
  value.reflect([&strings](auto &name, auto &value) {
    MString key{name};

    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
      motr::TagStr(key.hash, {value});
      // server->sendWebsocketText(value);
      strings.push_back(value);
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                        uint64_t>) {
      motr::TagInt(key.hash, {value});
    } else {
      MOTR_LOG("sendRPCResult unsupported type: field name={}", name);
    }
  });

  ServerOutboxString::send(strings);
}

template <typename T, typename>
std::optional<T> M::motr::RPC::getRPCResult(const EventTreeNode::Ptr &node) {
  uint64_t fingerprint = getFingerprint(node);
  if (fingerprint != T::RPCFingerprint)
    return std::nullopt;

  T result;
  std::unordered_map<Hash::Value, EventTreeNode *> children;
  for (auto &child : node->children) {
    children[child->message.id] = child.get();
  }

  result.reflect([&](auto &name, auto &value) {
    MString key{name};
    auto it = children.find(key.hash);
    if (it == children.end()) {
      MOTR_LOG("RPCResult[{}] child tag not found", name);
      return;
    }
    auto &child = it->second;
    Message &msg = child->message;
    uint64_t tagValue = msg.getTagValue();

    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
      if (msg.flags != MessageFlags::TagStr) {
        MOTR_LOG("RPCResult[{}] (unsupported type)", name);
        return;
      }
      value = MString{tagValue, false}.str(true);
    } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                        uint64_t>) {
      if (msg.flags != MessageFlags::TagInt) {
        MOTR_LOG("RPCResult[{}] (unsupported type)", name);
        return;
      }
      value = tagValue;
    }
  });
  return result;
}
#endif // MOTR_RPCMAILBOX_H
