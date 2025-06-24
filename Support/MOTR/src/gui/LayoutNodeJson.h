//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef MOTR_GUI_LAYOUT_NODE_JSON_H
#define MOTR_GUI_LAYOUT_NODE_JSON_H

#include "nlohmann/json_fwd.hpp"
#include <memory>

namespace M::motr::Gui {
struct LayoutNode;
enum class DVKind;

// Create a layout node from a JSON node
std::shared_ptr<LayoutNode>
makeLayoutNodeFromJsonNode(const nlohmann::json &jsonNode);

// Create a JSON node from a layout node
nlohmann::json makeJsonNodeFromLayoutNode(const LayoutNode *node);

} // namespace M::motr::Gui

#endif // MOTR_GUI_LAYOUT_NODE_JSON_H
