//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_EVENTTREE_H
#define MOTR_EVENTTREE_H

#include "motr/Log.h"
#include "motr/Message.h"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "motr/MString.h"
#include "motr/TagLibrary.h"

namespace M::motr {

struct TagLibrary;

struct EventTreeNode : std::enable_shared_from_this<EventTreeNode> {
  using Ptr = std::shared_ptr<EventTreeNode>;
  using WeakPtr = std::weak_ptr<EventTreeNode>;
  using Ptrs = std::vector<Ptr>;
  using WeakPtrs = std::vector<WeakPtr>;

  Message message;
  std::weak_ptr<EventTreeNode> parent;
  std::vector<Ptr> children;
  using size_t = std::size_t;

  std::shared_ptr<TagLibrary> tagLibrary;

  EventTreeNode(const Message &msg) : message(msg) {}
  ~EventTreeNode();

  std::shared_ptr<TagLibrary> getTagLibrary();

  EventTreeNode::Ptr getChildTagNode(MString key) const;

  template <typename T>
  std::optional<T> getChildTagValue(MString key) const {
    for (auto &child : children) {
      const Message &msg = child->message;
      if (msg.id == key.hash.v) {
        if (msg.flags == MessageFlags::TagStr) {
          if constexpr (std::is_same_v<T, MString>)
            return MString{msg.getTagValue(), false};
          if constexpr (std::is_same_v<T, std::string_view>)
            return MString{msg.getTagValue(), false}.sv(true);
          if constexpr (std::is_same_v<T, std::string>)
            return MString{msg.getTagValue(), false}.str(true);
          return std::nullopt;
        } else if (msg.flags == MessageFlags::TagInt) {
          if constexpr (std::is_same_v<T, uint64_t>)
            return msg.getTagValue();
          if constexpr (std::is_same_v<T, uint32_t>)
            return static_cast<uint32_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, uint16_t>)
            return static_cast<uint16_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, uint8_t>)
            return static_cast<uint8_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, int64_t>)
            return static_cast<int64_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, int32_t>)
            return static_cast<int32_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, int16_t>)
            return static_cast<int16_t>(msg.getTagValue());
          if constexpr (std::is_same_v<T, int8_t>)
            return static_cast<int8_t>(msg.getTagValue());
          return std::nullopt;
        }
        assert(false && "Invalid tag flags");
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  size_t getMemorySize() const {
    return sizeof(EventTreeNode) + sizeof(Ptr) * children.size() +
           sizeof(Message);
  }

  static size_t emptyMemorySize() {
    size_t childrenSize = sizeof(decltype(children));
    return sizeof(Message) + sizeof(decltype(parent));
  }

  std::vector<Ptr>
  getChildrenFiltered(std::function<bool(const Message &)> filter) const {
    std::vector<Ptr> result;
    for (const auto &child : children) {
      if (filter(child->message)) {
        result.push_back(child);
      }
    }
    return result;
  }

  std::vector<Ptr> getChildrenOfType(MessageType type) {
    return getChildrenFiltered(
        [type](const Message &msg) { return msg.type == type; });
  }

  std::vector<Ptr> getChildrenWith(MessageFlags flags) const {
    return getChildrenFiltered(
        [flags](const Message &msg) { return msg.flags == flags; });
  }

  std::vector<Ptr> getAncestors() const {
    std::vector<Ptr> ancestors;
    for (auto n = parent.lock(); n; n = n->parent.lock())
      ancestors.push_back(n);
    return ancestors;
  }

  void setParent(EventTreeNode *newParent) {
    if (auto curParent = this->parent.lock(); curParent) {
      curParent->children.erase(std::find(curParent->children.begin(),
                                          curParent->children.end(),
                                          shared_from_this()));
    }
    if (newParent) {
      this->parent = newParent->shared_from_this();
      newParent->children.push_back(shared_from_this());
    } else {
      this->parent.reset(); //= nullptr;
    }
  }

  size_t numAncestors() const {
    size_t count = 0;
    for (auto n = parent.lock(); n; n = n->parent.lock(), count++)
      ;
    return count;
  }

  Ptr getParent() const { return parent.lock(); }

  enum class TraverseMode {
    BreadthFirst,
    DFSPreOrder,
    DFSInOrder, // CAREFUL: this only works for binary trees
    DFSPostOrder
  };

  /*
  Example tree
      A
      ├── B
      │   ├── D
      │   └── E
      └── C
          ├── F
          └── G
  */

  // pre order: A, B, D, E, C, F, G
  std::vector<Ptr> getDFSPreOrder() {
    std::vector<Ptr> descendants;
    descendants.emplace_back(shared_from_this());
    for (auto &child : children) {
      auto childDescendants = child->getDFSPreOrder();
      descendants.insert(descendants.end(), childDescendants.begin(),
                         childDescendants.end());
    }
    return descendants;
  }

  // post order: D, E, B, F, G, C, A
  std::vector<Ptr> getDFSPostOrder() {
    std::vector<Ptr> descendants;
    for (auto &child : children) {
      auto childDescendants = child->getDFSPostOrder();
      descendants.insert(descendants.end(), childDescendants.begin(),
                         childDescendants.end());
    }
    descendants.emplace_back(shared_from_this());
    return descendants;
  }

  // in order: D, B, E, A, F, C, G
  std::vector<Ptr> getDFSInOrder() {
    assert(children.empty() ||
           children.size() == 2 && "DFSInOrder only works for binary trees");
    std::vector<Ptr> descendants;
    if (children.empty())
      return {shared_from_this()};

    if (children.size() == 2) {
      auto leftDescendants = children[0]->getDFSInOrder();
      auto rightDescendants = children[1]->getDFSInOrder();
      descendants.insert(descendants.end(), leftDescendants.begin(),
                         leftDescendants.end());
      descendants.emplace_back(shared_from_this());
      descendants.insert(descendants.end(), rightDescendants.begin(),
                         rightDescendants.end());
    }
    return descendants;
  }

  // breadth first: A, B, C, D, E, F, G
  std::vector<Ptr> getDescendantsBreadthFirst() {
    std::vector<Ptr> descendants;
    std::deque<Ptr> queue{shared_from_this()};
    while (!queue.empty()) {
      descendants.insert(descendants.end(), queue.begin(), queue.end());
      std::deque<Ptr> nextLevel;
      for (auto &child : queue)
        nextLevel.insert(nextLevel.end(), child->children.begin(),
                         child->children.end());
      std::swap(queue, nextLevel);
    }
    return descendants;
  }

  template <TraverseMode mode>
  std::vector<Ptr> getDescendants() {
    if constexpr (mode == TraverseMode::BreadthFirst)
      return getDescendantsBreadthFirst();
    if constexpr (mode == TraverseMode::DFSPreOrder)
      return getDFSPreOrder();
    if constexpr (mode == TraverseMode::DFSInOrder)
      return getDFSInOrder();
    if constexpr (mode == TraverseMode::DFSPostOrder)
      return getDFSPostOrder();
    return {};
  }

  std::vector<Ptr> getDescendants(TraverseMode mode) {
    switch (mode) {
    case TraverseMode::BreadthFirst:
      return getDescendants<TraverseMode::BreadthFirst>();
    case TraverseMode::DFSPreOrder:
      return getDescendants<TraverseMode::DFSPreOrder>();
    case TraverseMode::DFSInOrder:
      return getDescendants<TraverseMode::DFSInOrder>();
    case TraverseMode::DFSPostOrder:
      return getDescendants<TraverseMode::DFSPostOrder>();
    }
  }

  Ptr getPopEvent() const {
    if (message.flags == MessageFlags::Push) {
      // start from the end and go backwards
      // because the pop event is likely to be the last child
      for (size_t i = children.size(); i > 0; --i) {
        auto &c = children[i - 1];
        if (c->message.flags == MessageFlags::Pop) {
          return c;
        }
      }
    }
    return nullptr;
  }
  // returns 0 if this event is not a push event or if there is no pop event
  Time::Duration getDuration() const {
    if (auto popEvent = getPopEvent(); popEvent) {
      return {int64_t(popEvent->message.ts) - int64_t(message.ts)};
    }
    return {};
  }
};

struct EventTree {

  static EventTree &getSingleton() {
    static EventTree instance(1024 * 1024 * 1024);
    return instance;
  }

  // note: memory culling is not used or tested and may not work
  size_t maxMemorySize;
  size_t currentMemorySize;

  std::unordered_map<uint64_t, EventTreeNode *> nodeMap;

  EventTreeNode::Ptr root;

  std::unordered_map<uint64_t, std::vector<EventTreeNode::Ptr>> orphans;

  EventTree(size_t maxMemorySize)
      : maxMemorySize(maxMemorySize), currentMemorySize(0) {
    root = std::make_shared<EventTreeNode>(Message());
    root->message.flags = MessageFlags::Push;
    nodeMap[0] = root.get();
  }

  // Find an exact match for a message
  EventTreeNode *findMessage(const Message &msg) {
    if (auto iter = nodeMap.find(msg.id); iter != nodeMap.end()) {
      const Message &nodeMsg = iter->second->message;
      if (nodeMsg == msg)
        return iter->second;
      else {
        MOTR_LOG("{}", "Found conflicting messages with same id");
        // MOTR_LOG("Old msg: {}", toJSONString(nodeMsg));
        // MOTR_LOG("New msg: {}", toJSONString(msg));
      }
    }
    return nullptr;
  }

  EventTreeNode::Ptr addMessage(const Message &msg) {
    assert(msg.id != msg.pid && "Message id and pid cannot be the same");
    assert(msg.id != 0 && "Message id cannot be 0");

    // update memory size
    currentMemorySize += EventTreeNode::emptyMemorySize();

    auto node = std::make_shared<EventTreeNode>(msg);

    if (!msg.isaTag()) {
      // add non-tag nodes to node id->ptr map
      nodeMap[msg.id] = node.get();
    }

    {
      // note that pid==0 will find the root node as parent
      auto piter = nodeMap.find(msg.pid);

      // no parent
      if (piter == nodeMap.end()) {
        orphans[msg.pid].push_back(node);
      } else {
        node->setParent(piter->second); //->shared_from_this());
      }
    }

    // check if orphans have this node as a parent
    if (!msg.isaTag()) {
      if (auto iter = orphans.find(msg.id); iter != orphans.end()) {
        for (auto &orphan : iter->second) {
          orphan->setParent(node.get());
        }
        orphans.erase(iter);
      }
    }

    // disable untested memory culling
    // pruneIfNecessary();
    return node;
  }

  std::vector<EventTreeNode::Ptr> getAllNodes() const {
    std::vector<EventTreeNode::Ptr> allNodes;

    {
      auto rootNodes = root->getDFSPreOrder();
      allNodes.insert(allNodes.end(), rootNodes.begin(), rootNodes.end());
    }

    for (const auto &[pid, orphanRoots] : orphans) {
      for (auto &orphanRoot : orphanRoots) {
        auto allOrphanRootNodes = orphanRoot->getDFSPreOrder();
        allNodes.insert(allNodes.end(), allOrphanRootNodes.begin(),
                        allOrphanRootNodes.end());
      }
    }
    return allNodes;
  }

private:
  void pruneIfNecessary() {
    while (currentMemorySize > maxMemorySize) {
      pruneOldestNode();
    }
  }

  void pruneOldestNode() {
    assert(false && "Not implemented");
    /*
    if (roots.empty())
      return;

    nodeMap.erase(roots.begin()->second->message.id);
    roots.erase(roots.begin());
    */
  }
};

inline EventTreeNode::Ptr EventTreeNode::getChildTagNode(MString key) const {
  for (auto &child : children) {
    const Message &msg = child->message;
    if (msg.isaTag(false) && msg.id == key.hash.v) {
      return child;
    }
  }
  return nullptr;
}

inline EventTreeNode::~EventTreeNode() {
  if (!message.isaTag()) {
    auto &eventTree = EventTree::getSingleton();
    auto it = eventTree.nodeMap.find(message.id);
    if (it != eventTree.nodeMap.end()) {
      eventTree.nodeMap.erase(it);
    } else {
      MOTR_LOG("~EventTreeNode: Node not found in nodeMap: id={}", message.id);
    }
  }
}

} // namespace M::motr

#endif // MOTR_EVENTTREE_H
