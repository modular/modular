//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_ATTRIBUTERESOLVER_H
#define MOTR_ATTRIBUTERESOLVER_H

#include "DynamicValue.h"
#include "motr/Hash.h"

namespace M::motr {
class TagLibrary;
}

namespace M::motr::Gui {
class LayoutNode;
}

namespace M::motr::Gui::Attribute {
// Main resolution function for a node and all its children
// This resolves all dynamic attributes and updates nodes accordingly
void resolveNode(LayoutNode &node, M::motr::TagLibrary &tags);

} // namespace M::motr::Gui::Attribute

#endif // MOTR_ATTRIBUTERESOLVER_H
