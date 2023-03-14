//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrInterfaces.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace M;

//===----------------------------------------------------------------------===//
// HasAlignedBytesInterface for DenseResourceElementsAttr
//===----------------------------------------------------------------------===//

namespace {
struct DenseResourceElementsAttrHasAlignedBytesInterface
    : public HasAlignedBytesInterface::ExternalModel<
          DenseResourceElementsAttrHasAlignedBytesInterface,
          mlir::DenseResourceElementsAttr> {
  AlignedBytesType getAlignedBytesType(Attribute attr) const {
    auto denseAttr = dyn_cast<mlir::DenseResourceElementsAttr>(attr);
    if (auto alignedBytesType =
            llvm::dyn_cast<AlignedBytesType>(denseAttr.getType()))
      return alignedBytesType;
    mlir::AsmResourceBlob *blob = denseAttr.getRawHandle().getBlob();
    assert(blob && "dense_resource has not been initialized");
    uint64_t size = static_cast<uint64_t>(blob->getData().size());
    uint64_t align = static_cast<uint64_t>(blob->getDataAlignment());
    return AlignedBytesType::get(denseAttr.getContext(), size, align);
  }
};
} // namespace

void MDialect::injectAttrInterfaces() {
  mlir::DenseResourceElementsAttr::attachInterface<
      DenseResourceElementsAttrHasAlignedBytesInterface>(*getContext());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MAttrInterfaces.cpp.inc"
