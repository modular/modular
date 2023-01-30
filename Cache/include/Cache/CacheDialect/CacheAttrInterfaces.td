//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEDIALECT_CACHEATTRINTERFACES_TD
#define CACHE_CACHEDIALECT_CACHEATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ReplaceableAttrInterface
//===----------------------------------------------------------------------===//

def ReplaceableAttrInterface : AttrInterface<"ReplaceableAttr"> {
  let description = [{
    This interface describes an attribute that should be replaced when caching
    an IR region. This interface has a single method, `convertToIndex`, which
    allows the caching infrastructure to convert to something that looks like
    (or can be constructed from) and index.
  }];

  let cppNamespace = "::M::Cache";

  let methods = [
    InterfaceMethod<[{
      Given an index, return an attribute representing an index that could be
      used to reconstruct the attribute.
    }], "ReplaceableAttrIndex", "convertToIndex", (ins "size_t":$idx)>,
  ];
}

//===----------------------------------------------------------------------===//
// ReplaceableAttrIndexInterface
//===----------------------------------------------------------------------===//

def ReplaceableAttrIndexInterface : AttrInterface<"ReplaceableAttrIndex"> {
  let description = [{
    This interface describes an index-like attribute that can be used to index
    into a list of attributes. When the cache infrastructure elides attributes,
    it stores the elided attributes in a list on the RegionHashAttr. This
    interface provides the implementation to convert from an instance of this
    interface, given the list of attributes, to the original attribute.
  }];

  let cppNamespace = "::M::Cache";

  let methods = [
    InterfaceMethod<[{
      Given the attribute list, convert this index to the original attribute
      by indexing into the list.
    }], "ReplaceableAttr", "convertFromIndex",
        (ins "ArrayRef<Attribute>":$attrList)>,
  ];
}

#endif // CACHE_CACHEDIALECT_CACHEATTRINTERFACES_TD
