//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "EventTreeProcessor.h"
#include "GlobalState.h"
#include "LayoutLoader.h"
#include "motr/Log.h"
#include "motr/RPC.h"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

namespace M::motr {

const char SetLayoutCallFunctionName[] = "set_layout";

bool ProcessEventNode::is(EventTreeNode::Ptr node) {
  return node->message.type == MessageType::Process;
}

EventTreeNode::Ptr isaRPCCall(EventTreeNode::Ptr node) {
  if (!node)
    return nullptr;

  if (node->message.type != MessageType::RPCCall)
    return nullptr;

  if (node->message.flags != MessageFlags::Push)
    return nullptr;

  for (auto &child : node->children)
    if (child->message.type == MessageType::RPCCall &&
        child->message.flags == MessageFlags::Pop)
      return node;

  return nullptr;
}

TagLibrary::Ptr getTagsForRPCCall(EventTreeNode::Ptr node,
                                  std::string_view name) {
  node = isaRPCCall(node);
  if (!node)
    return nullptr;

  auto tags = node->getTagLibrary();
  if (tags->getString({Constants::__rpc_call_name__::hash}) != name)
    return nullptr;

  return tags;
}

template <typename T, const char *call_function_name>
bool CallEvent<T, call_function_name>::is(EventTreeNode::Ptr node) {
  return getTagsForRPCCall(node, getCallFunctionName()) != nullptr;
}

template <typename T, const char *call_function_name>
bool CallEvent<T, call_function_name>::hasAllTagStrs(
    EventTreeNode *node, std::initializer_list<std::string_view> tag_names) {
  if (!node)
    return false;
  for (auto &tag_name : tag_names) {
    if (!node->tagLibrary->hasTagStr({tag_name}))
      return false;
  }
  return true;
}

template <typename T, const char *call_function_name>
bool CallEvent<T, call_function_name>::hasAllTagInts(
    EventTreeNode *node, std::initializer_list<std::string_view> tag_names) {
  if (!node)
    return false;
  for (auto &tag_name : tag_names) {
    if (!node->tagLibrary->hasTagInt({tag_name}))
      return false;
  }
  return true;
}

bool SetLayoutEvent::is(EventTreeNode::Ptr node) {
  if (!CallEventType::is(node))
    return false;

  if (!hasAllTagStrs(node.get(), {"filename", "contents"}))
    return false;

  return true;
}

void handleProcessNode(EventTreeNode::Ptr node) {
  assert(node->message.type == MessageType::Process);
  auto &state = Gui::globalState();
  // todo: disabled for now because it's too slow
  return;
  auto allchildren =
      node->getDescendants<EventTreeNode::TraverseMode::DFSPreOrder>();
  for (auto &child : allchildren) {
    if (auto setLayoutEvent = SetLayoutEvent::as(child); setLayoutEvent) {
      auto filename = setLayoutEvent->get_filename();
      MOTR_LOG("set_layout filename={}", filename);
      Gui::loadLayoutJsonStringView(setLayoutEvent->get_contents());
    }
  }
}

template <int T>
bool checkGeneration() {
  auto &state = Gui::globalState();
  static int generation = 0;
  if (generation == state.generation)
    return false;
  generation = state.generation;
  return true;
}

// Explicit template instantiations
template bool checkGeneration<1>();
template bool checkGeneration<2>();

} // namespace M::motr
