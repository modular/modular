//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_MOTR_GUI_LAYOUT_LIBRARY_H
#define M_MOTR_GUI_LAYOUT_LIBRARY_H

#include "LayoutNode.h"
#include <string>
#include <unordered_map>

namespace M::motr::Gui {

struct LayoutLibrary {
  using LayoutMap = std::unordered_map<std::string, LayoutNode::Ptr>;

  LayoutNode::Ptr registerLayout(LayoutNode::Ptr node);
  LayoutNode::Ptr getLayout(const std::string &name) const;
  bool hasLayout(const std::string &name) const;
  void clear();

  static LayoutLibrary &instance();

  LayoutLibrary() = default;
  LayoutLibrary(const LayoutLibrary &) = delete;
  LayoutLibrary &operator=(const LayoutLibrary &) = delete;
  LayoutLibrary(LayoutLibrary &&) = delete;
  LayoutLibrary &operator=(LayoutLibrary &&) = delete;

  LayoutMap layouts;
};

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_LAYOUT_LIBRARY_H
