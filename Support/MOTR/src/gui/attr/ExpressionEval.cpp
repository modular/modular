//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ExpressionEval.h"
#include "DynamicValue.h"
#include "Parse.h"
#include "motr/TagLibrary.h"

namespace M::motr::Gui::Attribute::ExpressionEval {

// Check if a string contains variable references ${...}
bool containsVariables(std::string_view str) {
  // Simple check for ${
  return str.find("${") != std::string::npos;
}

// Find and evaluate variables within a string
std::string evaluate(std::string_view expr, TagLibrary &tags,
                     std::vector<MString> &deps) {
  if (!containsVariables(expr)) {
    return std::string{expr};
  }

  std::string result{expr};
  size_t startPos = 0;

  // Keep replacing variables until none left
  while (true) {
    // Find the next variable start
    size_t varStart = result.find("${", startPos);
    if (varStart == std::string::npos) {
      break; // No more variables
    }

    // Find matching end brace, handling nesting
    size_t pos = varStart + 2;
    int depth = 1;
    size_t varEnd = std::string::npos;

    while (pos < result.size() && depth > 0) {
      if (result[pos] == '{' && pos > 0 && result[pos - 1] == '$') {
        depth++;
      } else if (result[pos] == '}') {
        depth--;
        if (depth == 0) {
          varEnd = pos;
          break;
        }
      }
      pos++;
    }

    if (varEnd == std::string::npos) {
      // Unmatched open brace, skip it
      startPos = varStart + 2;
      continue;
    }

    // Extract variable name
    std::string varName = result.substr(varStart + 2, varEnd - varStart - 2);

    // Recursively evaluate nested variables in the variable name
    varName = evaluate(varName, tags, deps);

    // Store in the String Library
    MString varNameMStr{varName};

    // Add dependency
    deps.push_back(varNameMStr);

    // Look up in TagLibrary
    std::string_view replacement = tags.getString(varNameMStr);
    std::string tmp;
    if (replacement.empty()) {
      uint64_t asInt = 0;
      if (tags.getU64(varNameMStr, asInt)) {
        tmp = fmt::format("{}", asInt);
        replacement = tmp;
      } else {
        MOTR_LOG("ExpressionEval: variable {} not found", varName);
      }
    }

    // Replace in original string
    result.replace(varStart, varEnd - varStart + 1, replacement);

    // Continue search after replacement
    startPos = varStart + replacement.length();
  }

  return result;
}

} // namespace M::motr::Gui::Attribute::ExpressionEval
