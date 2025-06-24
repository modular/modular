//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RefLayoutNode.h"
#include "LayoutLibrary.h"
#include "imgui.h"
#include <yoga/Yoga.h>

namespace M::motr::Gui {

// Forward declaration of the template function
template <typename T>
static LayoutNode::Ptr cloneLayoutNode(const T *srcNode);

RefLayoutNode::RefLayoutNode(std::shared_ptr<LayoutNode> parent)
    : LayoutNode(parent) {
  type = LayoutNodeType::Ref;
}

void RefLayoutNode::draw(DrawContext &context) {
  setContextPosition(context, true);

  auto refNode = LayoutLibrary::instance().getLayout(refName);
  if (refNode) {
    DrawContext refContext = context;
    refNode->draw(refContext);
  } else {
    LayoutNode::draw(context);
  }
}

void RefLayoutNode::traverse(DrawContext &context) {
  draw(context);

  auto refNode = LayoutLibrary::instance().getLayout(refName);
  if (refNode) {
    ImVec2 childOffset = context.offset;
    childOffset.x += YGNodeLayoutGetLeft(node);
    childOffset.y += YGNodeLayoutGetTop(node);

    DrawContext refContext = context;
    refContext.offset = childOffset;
    refContext.depth++;

    for (auto &childLayoutNode : refNode->children) {
      DrawContext childContext = refContext;
      childLayoutNode->traverse(childContext);
    }
  } else {
    // Fall back to standard traversal if reference not found
    ImVec2 childOffset = context.offset;
    childOffset.x += YGNodeLayoutGetLeft(node);
    childOffset.y += YGNodeLayoutGetTop(node);

    for (auto &childLayoutNode : children) {
      DrawContext childContext = context;
      childContext.offset = childOffset;
      childContext.depth++;
      childLayoutNode->traverse(childContext);
    }
  }
}

} // namespace M::motr::Gui
