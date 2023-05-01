//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheAttrInterfaces.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace Cache;

namespace {
struct SymbolRefReplaceableAttr
    : public ReplaceableAttr::ExternalModel<SymbolRefReplaceableAttr,
                                            SymbolRefAttr> {
  /// Convert a SymbolRefAttr to an index by simply returning a symbol of the
  /// index, like `@"0"`.
  ReplaceableAttrIndex convertToIndex(Attribute attr, size_t idx) const {
    return cast<ReplaceableAttrIndex>(
        SymbolRefAttr::get(StringAttr::get(attr.getContext(), Twine(idx))));
  }
};

struct SymbolRefReplaceableAttrIndex
    : public ReplaceableAttrIndex::ExternalModel<SymbolRefReplaceableAttrIndex,
                                                 SymbolRefAttr> {
  /// Convert a SymbolRefAttr from its index representation and retrieve the
  /// attr from the list. Because we're using a janky representation where we
  /// have indices as symbols, we can't really tell when something is a
  /// SymbolRefAttr *as an index* vs when it's an actual symbol - so if
  /// conversion fails, we just roll with it.
  ReplaceableAttr convertFromIndex(Attribute attr,
                                   ArrayRef<Attribute> attrs) const {
    size_t index;
    bool err =
        cast<SymbolRefAttr>(attr).getLeafReference().getValue().getAsInteger(
            10, index);
    if (!err)
      return cast<ReplaceableAttr>(attrs[index]);
    return cast<ReplaceableAttr>(attr);
  }
};
} // namespace

void CacheDialect::injectAttrInterfaces() {
  SymbolRefAttr::attachInterface<SymbolRefReplaceableAttr,
                                 SymbolRefReplaceableAttrIndex>(*getContext());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheAttrInterfaces.cpp.inc"
