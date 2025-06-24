//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Style.h"

namespace M::motr::Gui {

// Style implementation
Style::Style(M::motr::Hash::Value name, Style *parent)
    : name(name), parent(parent) {}

void Style::setAttribute(M::motr::Hash::Value key, const DynamicValue &value) {
  attrs[key] = value;
}

const DynamicValue *Style::getAttribute(M::motr::Hash::Value key) const {
  // First check local attributes
  auto it = attrs.find(key);
  if (it != attrs.end()) {
    return &it->second;
  }

  // If not found locally, check parent chain
  if (parent) {
    return parent->getAttribute(key);
  }

  // Not found in entire chain
  return nullptr;
}

bool Style::hasLocalAttribute(M::motr::Hash::Value key) const {
  return attrs.find(key) != attrs.end();
}

void Style::clear() { attrs.clear(); }

// StyleLibrary implementation
StyleLibrary &StyleLibrary::instance() {
  static StyleLibrary instance;
  return instance;
}

Style *StyleLibrary::createStyle(const std::string &name, Style *parent) {
  Hash nameHash{name};
  auto style = std::make_unique<Style>(nameHash, parent);
  Style *result = style.get();
  styles[nameHash.v] = std::move(style);
  return result;
}

Style *StyleLibrary::getStyle(const std::string &name) const {
  Hash nameHash{name};
  auto it = styles.find(nameHash.v);
  if (it != styles.end()) {
    return it->second.get();
  }
  return nullptr;
}

void StyleLibrary::removeStyle(const std::string &name) {
  Hash nameHash{name};
  styles.erase(nameHash.v);
}

void StyleLibrary::clear() { styles.clear(); }

} // namespace M::motr::Gui
