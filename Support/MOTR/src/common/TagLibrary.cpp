//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/TagLibrary.h"
#include "motr/EventTree.h"
#include "motr/Log.h"
#include "motr/Message.h"
#include <cassert>

#include "motr/MString.h"
#include "motr/StringLibrary.h"

using namespace M::motr;

TagLibrary::TagLibrary() = default;

TagLibrary::~TagLibrary() = default;

namespace {
TagLibrary::TagIntMap getEventTreeNodeTagIntMap(const EventTreeNode &node) {
  TagLibrary::TagIntMap result;

  auto tagint_nodes = node.getChildrenWith(MessageFlags::TagInt);
  for (auto &tagint_node : tagint_nodes) {
    auto &msg = tagint_node->message;
    result[{msg.id}] = msg.getTagValue();
  }
  return result;
}

TagLibrary::TagStrMap getEventTreeNodeTagStrMap(const EventTreeNode &node) {
  TagLibrary::TagStrMap result;
  auto tagNodes = node.getChildrenWith(MessageFlags::TagStr);
  for (auto &tagNode : tagNodes) {
    auto &msg = tagNode->message;
    result[{msg.id}] = MString{msg.getTagValue(), false}.str(true);
  }
  return result;
}
} // namespace

size_t TagLibrary::initFromEventTreeNode(const EventTreeNode &node) {
  tagStrMap = getEventTreeNodeTagStrMap(node);
  tagIntMap = getEventTreeNodeTagIntMap(node);
  for (auto &[key, value] : tagStrMap) {
    tagVersionMap[key] = 1;
  }
  for (auto &[key, value] : tagIntMap) {
    tagVersionMap[key] = 1;
  }
  return tagStrMap.size() + tagIntMap.size();
}

TagLibrary::Ptr TagLibrary::create(const EventTreeNode &node) {
  auto tagLibrary = std::make_shared<TagLibrary>();
  tagLibrary->initFromEventTreeNode(node);
  if (auto parent = node.getParent()) {
    tagLibrary->parentContext = parent->getTagLibrary();
  }
  return tagLibrary;
}

TagLibrary::Ptr TagLibrary::pushContext(Ptr parent) {
  auto child = std::make_shared<TagLibrary>();
  child->parentContext = parent;
  return child;
}

bool TagLibrary::getString(MString key, std::string_view &value) const {
  if (auto it = tagStrMap.find(key); it != tagStrMap.end()) {
    value = it->second;
    return true;
  }

  if (!isLocalOnly()) {
    if (auto parent = parentContext.lock(); parent != nullptr) {
      return parent->getString(key, value);
    }
  }
  return false;
}

std::optional<std::string_view>
TagLibrary::getOptionalString(MString key) const {
  std::string_view value;
  if (getString(key, value))
    return value;
  return {};
}

std::string_view TagLibrary::getString(MString key) const {
  std::string_view value;
  if (getString(key, value))
    return value;
  return {};
}

MString TagLibrary::setString(MString key, std::string_view val) {

  // if the value is the same, dont do anything
  {
    std::string_view oldVal;
    if (getString(key, oldVal))
      if (oldVal == val)
        return key;
  }
  tagStrMap[key] = std::string(val);
  tagVersionMap[key]++;
  return key;
}

bool TagLibrary::hasTagStr(MString key) const {
  if (tagStrMap.find(key) != tagStrMap.end())
    return true;

  if (!isLocalOnly()) {
    if (auto parent = parentContext.lock(); parent != nullptr) {
      return parent->hasTagStr(key);
    }
  }

  return false;
}

bool TagLibrary::hasTagInt(MString key) const {
  if (tagIntMap.find(key) != tagIntMap.end())
    return true;

  if (!isLocalOnly()) {
    if (auto parent = parentContext.lock(); parent != nullptr) {
      return parent->hasTagInt(key);
    }
  }

  return false;
}

bool TagLibrary::hasTag(MString key) const {
  return hasTagStr(key) || hasTagInt(key);
}

bool TagLibrary::getU64(MString key, uint64_t &value) const {
  auto it = tagIntMap.find(key);
  if (it != tagIntMap.end()) {
    value = it->second;
    return true;
  }
  if (!isLocalOnly()) {
    if (auto parent = parentContext.lock(); parent != nullptr) {
      return parent->getU64(key, value);
    }
  }

  return false;
}

std::optional<uint64_t> TagLibrary::getOptionalU64(MString key) const {
  uint64_t value = 0;
  if (getU64(key, value))
    return value;
  return {};
}

uint64_t TagLibrary::getU64(MString key) const {
  uint64_t value = 0;
  if (getU64(key, value))
    return value;
  return 0;
}

MString TagLibrary::setU64(MString key, uint64_t value) {
  {
    // if the value is the same, dont do anything
    uint64_t oldValue = 0;
    if (getU64(key, oldValue))
      if (oldValue == value)
        return key;
  }

  tagIntMap[key] = value;
  tagVersionMap[key]++;
  return key;
}

std::shared_ptr<TagLibrary> EventTreeNode::getTagLibrary() {
  if (!tagLibrary) {
    tagLibrary = TagLibrary::create(*this);
  }
  return tagLibrary;
}

uint64_t TagLibrary::getVersion(MString key) const {
  auto it = tagVersionMap.find(key);
  if (it != tagVersionMap.end()) {
    return it->second;
  }
  return 0;
}
