//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_YOGAAPPLY_H
#define MOTR_YOGAAPPLY_H

#include "motr/MString.h"

namespace M::motr::Gui {
struct LayoutNode;
}

namespace M::motr::Gui::YogaApply {

// Apply a resolved dynamic value to a yoga node based on attribute key
bool applyDynamicValue(LayoutNode &node, MString key);
} // namespace M::motr::Gui::YogaApply

#endif // MOTR_YOGAAPPLY_H
