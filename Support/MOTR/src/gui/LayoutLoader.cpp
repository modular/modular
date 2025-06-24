//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LayoutLoader.h"
#include "GlobalState.h"
#include "LayoutNodeJson.h"
#include "WebUtilities.h"
#include "attr/AttributeResolver.h"
#include "motr/Log.h"
#include "motr/StringLibrary.h"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <unordered_set>

namespace M::motr::Gui {

void loadLayoutJsonStringView(std::string_view jsonString) {
  auto &state = globalState();
  std::string jsonStripped = strip_comments_sv(jsonString);
  auto layoutNode = LayoutNode::makeFromJsonStrView(jsonStripped);

  auto layoutWindow = WindowLayoutNode::wrap(layoutNode);

  Attribute::resolveNode(*layoutWindow, state.getTagLibrary());

  auto &name = layoutWindow->get_name();
  layoutWindow->visible = name != "RootView";
  state.layoutWindows[name] = layoutWindow;
  state.getLayoutLibrary().registerLayout(layoutNode);
}

LayoutNode::Ptr getLayoutNamed(const std::string &name) {
  auto &layoutLibrary = globalState().getLayoutLibrary();
  auto layout = layoutLibrary.getLayout(name);
  static std::unordered_set<std::string> seen;
  if (!layout) {
    if (seen.find(name) != seen.end())
      return nullptr;
    seen.insert(name);
    auto url = fmt::format("layouts/{}.json", name);
    auto onload = [](std::string_view data_sv) {
      loadLayoutJsonStringView(data_sv);
    };
    fetchUrlAsync(url, onload, nullptr);
  }
  return layout;
}

} // namespace M::motr::Gui

extern "C" {
EMSCRIPTEN_KEEPALIVE void handleLayoutJson(const char *jsonString) {
  M::motr::Gui::loadLayoutJsonStringView(jsonString);
}
}
