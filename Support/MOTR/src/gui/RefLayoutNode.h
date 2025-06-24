//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_REF_LAYOUT_NODE_H
#define M_MOTR_GUI_REF_LAYOUT_NODE_H

#include "LayoutNode.h"
#include <memory>
#include <string>

namespace M::motr::Gui {

// Forward declaration to avoid circular dependency
class LayoutLibrary;

struct RefLayoutNode : public LayoutNode {
  using Ptr = std::shared_ptr<RefLayoutNode>;
  RefLayoutNode(std::shared_ptr<LayoutNode> parent);
  void draw(DrawContext &context) override;
  void traverse(DrawContext &context) override;
  LayoutNode::Ptr clone() const override;

  std::string refName;
};

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_REF_LAYOUT_NODE_H
