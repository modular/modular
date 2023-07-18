//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"

#include "KGEN/POPDialect/POPTypes.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

static StructDeclOp createStruct(FileModuleOp module, StringAttr nameAttr,
                                 SmallVector<TypeAttr> const &fields,
                                 Location location) {
  OpBuilder b(module.getRegion());
  StructDeclOp declOp = b.create<StructDeclOp>(location, nameAttr);
  if (declOp.getFields().empty())
    declOp.getFields().push_back(new Block());
  b.setInsertionPointToStart(&declOp.getFields().front());
  unsigned i = 0;
  for (TypeAttr type : fields)
    b.create<StructFieldOp>(
        location,
        StringAttr::get(b.getContext(), "field" + std::to_string(i++)), type,
        nullptr);
  return declOp;
}

StructDeclOp ClosureEmitter::createClosureWrapperStructDecl(
    StringAttr name, Location location, SignatureType signatureType) {
  auto emptyList =
      POP::ArrayType::get(0, IntegerType::get(fileModuleOp.getContext(), 1));
  auto opaquePointer = TypeAttr::get(POP::PointerType::get(emptyList));
  SmallVector<TypeAttr> fieldTypes;
  fieldTypes.push_back(opaquePointer);
  StructDeclOp declOp = createStruct(fileModuleOp, name, fieldTypes, location);
  TypedAttr signatureAttr = SymbolConstantAttr::get(
      SymbolRefAttr::get(
          StringAttr::get(name.getContext(), name.str() + "_closureSignature")),
      signatureType);
  declOp.setClosureSignatureAttr(signatureAttr);
  return declOp;
}
