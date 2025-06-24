//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_ATTR_CONVERT_H
#define MOTR_ATTR_CONVERT_H

#include <string>

namespace M::motr::Gui::Attribute {

struct DynamicValue;

bool convertValue(std::string_view val_sv, DynamicValue &dv);

} // namespace M::motr::Gui::Attribute

#endif // MOTR_ATTR_CONVERT_H
