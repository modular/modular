//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_STYLE_H
#define MOTR_STYLE_H

#include "DynamicValue.h"
#include "motr/Hash.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace M::motr::Gui {

// Style class - holds a named collection of attributes with optional parent
// inheritance
class Style {
public:
  // Create a style with optional parent
  Style(M::motr::Hash::Value name, Style *parent = nullptr);

  // Get the name of this style
  M::motr::Hash::Value getName() const { return name; }

  // Get parent style (may be nullptr)
  Style *getParent() const { return parent; }

  // Set parent style
  void setParent(Style *newParent) { parent = newParent; }

  // Add/replace an attribute in this style
  void setAttribute(M::motr::Hash::Value key, const DynamicValue &value);

  // Get an attribute, searching parent chain if not found locally
  // Returns nullptr if not found in entire chain
  const DynamicValue *getAttribute(M::motr::Hash::Value key) const;

  // Check if this style contains a specific attribute (without checking parent)
  bool hasLocalAttribute(M::motr::Hash::Value key) const;

  // Clear all attributes
  void clear();

  // Get number of local attributes
  size_t size() const { return attrs.size(); }

  // Iterator support for local attributes only
  using AttributeMap = std::unordered_map<M::motr::Hash::Value, DynamicValue>;
  AttributeMap::const_iterator begin() const { return attrs.begin(); }
  AttributeMap::const_iterator end() const { return attrs.end(); }

private:
  M::motr::Hash::Value name; // Style name
  Style *parent;             // Parent style (nullptr if none)
  AttributeMap attrs;        // Local attributes
};

// StyleLibrary - manages styles and their relationships
class StyleLibrary {
public:
  // Get singleton instance
  static StyleLibrary &instance();

  // Create a new style with optional parent
  Style *createStyle(const std::string &name, Style *parent = nullptr);

  // Get a style by name, returns nullptr if not found
  Style *getStyle(const std::string &name) const;

  // Remove a style
  void removeStyle(const std::string &name);

  // Clear all styles
  void clear();

private:
  StyleLibrary() = default;
  ~StyleLibrary() = default;

  // Storage for styles
  std::unordered_map<M::motr::Hash::Value, std::unique_ptr<Style>> styles;
};

} // namespace M::motr::Gui

#endif // MOTR_STYLE_H
