//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_LAYOUT_LOADER_H
#define M_MOTR_GUI_LAYOUT_LOADER_H

#include "LayoutNode.h"
#include <string_view>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace M::motr::Gui {

void loadLayoutJsonStringView(std::string_view jsonString);
LayoutNode::Ptr getLayoutNamed(const std::string &name);

} // namespace M::motr::Gui

extern "C" {
EMSCRIPTEN_KEEPALIVE void handleLayoutJson(const char *jsonString);
}

#endif // M_MOTR_GUI_LAYOUT_LOADER_H
