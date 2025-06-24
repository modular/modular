//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_ATTR_EXPRESSIONEVAL_H
#define MOTR_ATTR_EXPRESSIONEVAL_H

#include "DynamicValue.h"
#include "motr/Hash.h"
#include <string>
#include <vector>

namespace M::motr {
class TagLibrary;
}

namespace M::motr::Gui::Attribute::ExpressionEval {

// Evaluate an expression with variable substitution
// Returns the evaluated string and populates deps with dependencies
std::string evaluate(std::string_view expr, //
                     TagLibrary &tags,      //
                     std::vector<MString> &deps);

} // namespace M::motr::Gui::Attribute::ExpressionEval

#endif // MOTR_EXPRESSIONEVAL_H
