//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_TAGS_H
#define MOTR_TAGS_H

#include "motr/Constants.h"
#include "motr/Hash.h"
#include "motr/Macros.h"
#include "motr/Mailbox.h"
#include "motr/Message.h"
#include "motr/motr.h"

#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace M::motr {

// TagStr is a special key value pair message that is sent to the server
// It only sends key/value hashes, not the strings
struct TagStr {
  explicit TagStr(Hash::Value key, Hash::Value value);
};

// TagInt is a special key value pair message that is sent to the server
// It only sends key hashes, not the string
struct TagInt {
  using Int = uint64_t;
  explicit TagInt(Hash::Value key, Int value);
};

// special case of TagStr
// that only sends a key/value ONCE per
// template instantiation of key/value
template <uint64_t key, uint64_t value>
struct TagStrOnce {
  TagStrOnce();
};

MOTR_ALWAYS_INLINE TagStr::TagStr(Hash::Value key, Hash::Value value) {
  // TagStr and TagInt are special cases of Message
  // layout that do not conform to the other Message types
  // todo: Use a more strictly enforced Message use pattern
#if 0
  static std::unordered_set<uint64_t> sent;
  uint64_t cache_key = key.v ^ value.v;
  if(sent.find(cache_key) != sent.end())
    return;
  sent.insert(cache_key);
#endif

  ServerOutbox::send({
      MessageType::Set,     // type
      MessageFlags::TagStr, // flags
      {},                   // data
      0,                    // procid
      value.v,              // ts
      key.v,                // id
      getParentID(),        // parentid
  });
}

MOTR_ALWAYS_INLINE TagInt::TagInt(Hash::Value key, Int value) {
  ServerOutbox::send({
      MessageType::Set,     // type
      MessageFlags::TagInt, // flags
      {},                   // data
      0,                    // procid
      value,                // ts
      key.v,                // id
      getParentID(),        // parentid
  });
}

template <uint64_t key, uint64_t value>
MOTR_ALWAYS_INLINE TagStrOnce<key, value>::TagStrOnce() {
  static std::once_flag once;
  std::call_once(once, []() {
    ServerOutbox::send({
        MessageType::Set,
        MessageFlags::TagStr,
        {},
        getProcessID(),
        value,
        key,
        getParentID(),
    });
  });
}

// todo: reverse these parameters
template <typename T, uint64_t key>
struct TagIntOnExit {
  TagIntOnExit() = default;
  // use destructor to send the tag
  ~TagIntOnExit();

  // no copy, allow move
  TagIntOnExit(const TagIntOnExit &) = delete;
  TagIntOnExit &operator=(const TagIntOnExit &) = delete;
  TagIntOnExit(TagIntOnExit &&) = default;
  TagIntOnExit &operator=(TagIntOnExit &&) = default;

  T value = {};
  TagIntOnExit &operator=(T value);
  operator T() const;
};

template <typename T, uint64_t key>
MOTR_ALWAYS_INLINE TagIntOnExit<T, key> &
TagIntOnExit<T, key>::operator=(T value) {
  this->value = value;
  return *this;
}

template <typename T, uint64_t key>
MOTR_ALWAYS_INLINE TagIntOnExit<T, key>::operator T() const {
  return value;
}

template <typename T, uint64_t key>
MOTR_ALWAYS_INLINE TagIntOnExit<T, key>::~TagIntOnExit() {
  ServerOutbox::send({
      MessageType::Set,
      MessageFlags::TagInt,
      {},
      getProcessID(),
      value,
      key,
      getParentID(),
  });
}

struct SendTags {
  using StringString = std::pair<const char *, const char *>;
  using TagStringList = std::vector<StringString>;

  using StringInt = std::pair<const char *, uint64_t>;
  using TagIntList = std::vector<StringInt>;

  SendTags(const TagStringList &tags);
  SendTags(const TagIntList &tags);
};

MOTR_ALWAYS_INLINE SendTags::SendTags(const TagStringList &tags) {
  for (const auto &[key, value] : tags) {
    TagStr{Hash::Value{key}, Hash::Value{value}};
  }
}

MOTR_ALWAYS_INLINE SendTags::SendTags(const TagIntList &tags) {
  for (const auto &[key, value] : tags) {
    Hash::Value keyHash{key};
    TagInt{keyHash, value};
  }
}

} // namespace M::motr

#endif
