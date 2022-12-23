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
      "Optional<int64_t>", "getTypeSize", (ins "TargetInfoAttr":$target)>,
    InterfaceMethod<
      /*desc=*/"Get the alignment in bytes of the type for the given target.",
      "Optional<int64_t>", "getTypeAlign", (ins "TargetInfoAttr":$target)>,
  ];

  let extraClassDeclaration = [{
    /// Try to get the size in bytes of the type.
    static Optional<int64_t>
    getTypeSizeInBytes(TargetInfoAttr target, Type type);

    /// Try to get the aligment in bytes of the type.
    static Optional<int64_t>
    getTypeAlignInBytes(TargetInfoAttr target, Type type);
  }];
}

#endif // MTYPEINTERFACES_TD
