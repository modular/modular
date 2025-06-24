//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DragAndDrop.h"
#include "GlobalState.h"
#include "ModularTraceSpan.h"
#include "motr/EventTree.h"
#include "motr/JSON.h" // Please include first to hook up the error handler
#include "motr/Log.h"
#include "motr/Message.h"
// https://github.com/biojppm/rapidyaml/blob/d8f4d0150b3f84dca5d2775154f224e0c580abc5/README.md#single-header-file
#define RYML_SINGLE_HDR_DEFINE_NOW
#include <rapidyaml-0.7.2.hpp>
#include <string_view>
#include <vector>

using namespace M;

extern std::vector<std::string> messages;

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html#access-memory-from-javascript
extern "C" {

EMSCRIPTEN_KEEPALIVE void handleJsonData(const char *jsonData) {
  parseJsonData(jsonData);
}

EMSCRIPTEN_KEEPALIVE void handleYamlData(const char *yamlData) {
  parseYamlData(yamlData);
}
}
#endif

void parseJsonData(const char *jsonData) {
  nlohmann::json json = nlohmann::json::parse(jsonData);

  for (const auto &[key, value] : json.items()) {
    printf("JSON Parsed key [%s], value omitted\n", key.c_str());
  }

  if (!json.contains("traceEvents")) {
    printf("JSON does not contain traceEvents\n");
    return;
  }
  auto traceEvents = json["traceEvents"];

  /*
    auto& state = M::motr::Gui::globalState();
    std::vector<motr::Gui::ModularTraceSpan> &spans = state.spans;
    spans.clear();
    for (const auto &traceEvent : traceEvents) {
      if (traceEvent.contains("ph") && traceEvent["ph"] == "X") {
        spans.emplace_back(traceEvent);
      }
    }
    */
}

void parseYamlData(const char *yamlData) {
  ryml::Tree tree = ryml::parse_in_arena(c4::csubstr(yamlData));
  auto root = tree.rootref();
  if (root.is_map()) {
    for (const auto &item : root.children()) {
      const auto &key = item.key();
      printf("YAML Parsed key [%s], value omitted\n",
             std::string(key.str, key.len).c_str());
    }
  }
}
