//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MDIALECT_MATTRINTERFACES_TD
#define SUPPORT_MDIALECT_MATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// HasAlignedBytesInterface
//===----------------------------------------------------------------------===//

def HasAlignedBytesInterface : AttrInterface<"HasAlignedBytesInterface"> {
  let description = [{
    This interface allows an attribute to describe the size and
    alignment of its underlying ArrayRef<uint8_t> data as an !M.aligned_bytes.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<
      [{
        Returns the M::AlignedBytesType describing the size and alignment of an
        attribute.
      }],
      "AlignedBytesType",
      "getAlignedBytesType",
      (ins )>,
  ];
}

#endif  // SUPPORT_MDIALECT_MATTRINTERFACES_TD
