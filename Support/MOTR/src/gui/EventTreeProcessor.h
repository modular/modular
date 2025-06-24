//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_EVENT_TREE_PROCESSOR_H
#define M_MOTR_GUI_EVENT_TREE_PROCESSOR_H

#include "motr/EventTree.h"
#include "motr/Hash.h"
#include "motr/MString.h"
#include "motr/TagLibrary.h"
#include <initializer_list>
#include <memory>
#include <string_view>

#define define_get_tag_string_view(tag_name)                                   \
  std::string_view get_##tag_name() const {                                    \
    constexpr auto key = M::motr::Hash::Value{#tag_name};                      \
    return tagLibrary->getString(key);                                         \
  }

#define define_get_tag_u64(tag_name)                                           \
  uint64_t get_##tag_name() const {                                            \
    constexpr auto key = M::motr::Hash::Value{#tag_name};                      \
    return tagLibrary->getU64(key);                                            \
  }

namespace M::motr {

template <typename T>
struct TypedEventTreeNode : EventTreeNode {
  using Type = T;
  using Ptr = std::shared_ptr<Type>;
  using WeakPtr = std::weak_ptr<Type>;
  using Ptrs = std::vector<Ptr>;
  using WeakPtrs = std::vector<WeakPtr>;

  static Ptr as(EventTreeNode::Ptr node) {
    if (T::is(node))
      return std::static_pointer_cast<Type>(node);
    return nullptr;
  }
};

struct ProcessEventNode : TypedEventTreeNode<ProcessEventNode> {
  static bool is(EventTreeNode::Ptr node);
  define_get_tag_u64(ProcessId);
};

template <typename T, const char *call_function_name>
struct CallEvent : TypedEventTreeNode<T> {
  using CallEventType = CallEvent<T, call_function_name>;

  static constexpr std::string_view getCallFunctionName() {
    return call_function_name;
  }

  static bool is(EventTreeNode::Ptr node);
  static bool hasAllTagStrs(EventTreeNode *node,
                            std::initializer_list<std::string_view> tag_names);
  static bool hasAllTagInts(EventTreeNode *node,
                            std::initializer_list<std::string_view> tag_names);
};

extern const char SetLayoutCallFunctionName[];
struct SetLayoutEvent : CallEvent<SetLayoutEvent, SetLayoutCallFunctionName> {
  static bool is(EventTreeNode::Ptr node);
  define_get_tag_string_view(filename);
  define_get_tag_string_view(contents);
};

// Processing functions
void handleProcessNode(EventTreeNode::Ptr node);
EventTreeNode::Ptr isaCall(EventTreeNode::Ptr node);
TagLibrary::Ptr getTagsForCall(EventTreeNode::Ptr node, std::string_view name);

template <int T>
bool checkGeneration();

} // namespace M::motr

#endif // M_MOTR_GUI_EVENT_TREE_PROCESSOR_H
