//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LayoutLibrary.h"
#include "LayoutNodeJson.h"
#include "motr/Log.h"
#include <yoga/Yoga.h>
namespace M::motr::Gui {

LayoutNode::Ptr LayoutLibrary::registerLayout(LayoutNode::Ptr node) {
  // assert(!hasLayout(node->name) && "Layout already registered");
  const auto &name = node->get_name();
  MOTR_LOG("Registering layout: {}", name);
  layouts[name] = node;
  return node;
}

LayoutNode::Ptr LayoutLibrary::getLayout(const std::string &name) const {
  auto it = layouts.find(name);
  if (it != layouts.end())
    return it->second;
  return nullptr;
}

bool LayoutLibrary::hasLayout(const std::string &name) const {
  return layouts.find(name) != layouts.end();
}

void LayoutLibrary::clear() { layouts.clear(); }

LayoutLibrary &LayoutLibrary::instance() {
  static LayoutLibrary instance;
  return instance;
}

} // namespace M::motr::Gui
