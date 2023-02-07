//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MTYPEINTERFACES_TD
#define MTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

def DataLayoutInterface : TypeInterface<"DataLayoutInterface"> {
  let description = [{
    This interfaces allows the data layout of a type to be queried. It provides
    methods to query the size and alignment of the type. These methods are
    defined to be failable so that aggregate types may return null if a
    contained type does not implement the interface.

    The data layouts of types is target-dependent, which is why the methods
    require a target configuration to be passed. Callers should use the static
    `getTypeSizeInBytes` and `getTypeAlignInBytes` methods.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the size in bytes of the type for the given target.",
      "std::optional<int64_t>", "getTypeSize", (ins "TargetInfoAttr":$target)>,
    InterfaceMethod<
      /*desc=*/"Get the alignment in bytes of the type for the given target.",
      "std::optional<int64_t>", "getTypeAlign", (ins "TargetInfoAttr":$target)>,
  ];

  let extraClassDeclaration = [{
    /// Get the store size of the type. This is the maximum number of bytes that
    /// will be overwritten by a store of this type. It is equal to the type
    /// bitwidth rounded up to the nearest byte.
    static std::optional<int64_t>
    getTypeStoreSize(TargetInfoAttr target, Type type);
    /// Get the alloc size of the type. This is the number of bytes including
    /// padding that are occupied by one element of this type.
    static std::optional<int64_t>
    getTypeAllocSize(TargetInfoAttr target, Type type);

    /// Get the ABI alignment of the type in bytes.
    static std::optional<int64_t>
    getTypeABIAlign(TargetInfoAttr target, Type type);
  }];
}

#endif // MTYPEINTERFACES_TD
