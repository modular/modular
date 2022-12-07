//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

def KGEN_DataLayoutInterface : TypeInterface<"DataLayoutInterface"> {
  let description = [{
    This interface is used to query the size and layout of types.
  }];

  let cppNamespace = "::M::KGEN";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the size in bytes of the type.",
      "Optional<int64_t>", "getTypeSize", (ins "TargetInfoAttr":$target)>,
    InterfaceMethod<
      /*desc=*/"Get the alignment in bytes of the type.",
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

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_TD
