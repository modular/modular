//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AttributeResolver.h"
#include "../LayoutNode.h"
#include "Convert.h"
#include "ExpressionEval.h"
#include "YogaApply.h"
#include "motr/MString.h"
#include "motr/StringLibrary.h"
#include "motr/TagLibrary.h"
#include <algorithm>
#include <string>

namespace M::motr::Gui::Attribute {

// Check if dependencies in DV have changed in TagLibrary
static bool needsUpdate(DynamicValue &dv, M::motr::TagLibrary &tags) {

  // index 0 is Unresolved, meaning the cache is empty
  // and we need to resolve
  if (dv.cache.index() == 0)
    return true;

  // If no dependencies, no update needed
  if (dv.deps.empty())
    return false;

  // If version vectors have different sizes, needs update
  if (dv.deps.size() != dv.versions.size())
    return true;

  // Check if any dependency has a different version
  for (size_t i = 0; i < dv.deps.size(); i++) {
    if (tags.getVersion(dv.deps[i]) != dv.versions[i])
      return true;
  }

  return false; // For now, assume no changes
}

// Resolve a single attribute value based on kind
[[nodiscard]] static bool resolveAttribute(DynamicValue &dv,
                                           M::motr::TagLibrary &tags) {
  // Skip if doesn't need update
  if (!needsUpdate(dv, tags))
    return false;

  std::string_view exprSV = dv.expr.sv();

  // Handle expression with variables
  dv.deps.clear();
  std::string evaluated =
      Attribute::ExpressionEval::evaluate(exprSV, tags, dv.deps);

  // update version info
  dv.versions.resize(dv.deps.size(), 0);
  for (size_t i = 0; i < dv.deps.size(); i++) {
    dv.versions[i] = tags.getVersion(dv.deps[i]);
  }

  bool debug = false;
  // debug = true;
  if (debug) {
    std::string depsStr;
    for (auto &dep : dv.deps) {
      depsStr += dep.str() + ", ";
    }
    depsStr = depsStr.substr(0, depsStr.size() - 2);

    MOTR_LOG("resolve: expr={} -> {}, with deps={}", exprSV, evaluated,
             depsStr);
  }

  return Attribute::convertValue(evaluated, dv);
}

// Main resolution function: updates all dynamic attributes
void resolveNode(LayoutNode &node, TagLibrary &tags) {

  // Process each attribute
  for (auto &[key, dynamicValue] : node.attrs) {
    if (!resolveAttribute(dynamicValue, tags))
      continue;

    // MOTR_LOG("resolved: key={}, val={}", key.sv(), dv.raw.sv() );
    YogaApply::applyDynamicValue(node, key);
  }

  // Recursively process children
  for (auto &child : node.children) {
    resolveNode(*child, tags);
  }
}

} // namespace M::motr::Gui::Attribute
