//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_WEB_UTILITIES_H
#define M_MOTR_GUI_WEB_UTILITIES_H

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/fetch.h>
#endif

#include "motr/JSON.h"

namespace M::motr::Gui {

// URL parameter parsing
std::unordered_map<std::string, std::string> &getURLSearchParams();
int getWebsocketPort(int idx);

// JavaScript execution
std::string eval(std::string_view cmd);

// Async fetch operations
void fetchUrlAsync(const std::string &url,
                   std::function<void(std::string_view)> onSuccess,
                   std::function<void(int, const char *)> onError);

void asyncLoadJsonUrl(std::string_view url,
                      std::function<bool(nlohmann::json)> onSuccess);

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_WEB_UTILITIES_H
