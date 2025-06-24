//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_ATTR_PARSE_H
#define MOTR_ATTR_PARSE_H

#include "DynamicValue.h"
#include <string>
#include <string_view>

namespace M::motr::Gui::Color {
struct RGBA32;
}

namespace M::motr::Gui::Attribute::Parse {

// Parse width/height values with unit detection
bool parseDimensionValue(std::string_view str, DimensionResolved &result);
bool parseColor(std::string_view str, Color::RGBA32 &result);
bool parseFlexResolved(std::string_view str, FlexResolved &result);

} // namespace M::motr::Gui::Attribute::Parse

#endif // MOTR_ATTR_PARSE_H
