//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoTypeDataLayout.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

//===----------------------------------------------------------------------===//
// MojoTypeDataLayoutContext::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeDataLayoutContext::Impl {
  Impl(MojoParserContext &context, TargetInfoAttr targetInfo)
      : context(context), targetInfo(targetInfo) {}

  /// Implementation of `MojoTypeDataLayoutContext::getOrCalculate`
  const std::optional<MojoTypeDataLayout> &
  getOrCalculate(MojoASTTypeRef typeRef);

  /// Calculate the data layout of the given type.
  std::optional<MojoTypeDataLayout> calculate(MojoASTTypeRef type);

  /// Calculate the data layout of the given struct.
  std::optional<MojoTypeDataLayout>
  calculateForStruct(MojoASTTypeRef type, LIT::StructDeclOp structOp);

  DenseMap<const void *, std::optional<MojoTypeDataLayout>> cache;
  MojoParserContext &context;
  TargetInfoAttr targetInfo;
};

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStruct(
    MojoASTTypeRef typeRef, LIT::StructDeclOp structOp) {
  auto refType = cast<DeclRefType>(typeRef);
  uint64_t size = 0, align = 1;
  MojoTypeDataLayout layout;

  for (LIT::StructFieldOp field : structOp.getFieldDecls()) {
    MojoASTTypeRef fieldType = MojoASTTypeRef(field.getTypeAttr().getValue());

    // If the DeclRefType has parameters, try to evaluate and substitute them
    // into the type.
    if (!refType.getParamValues().empty())
      fieldType = context.concretizeType(refType.getParamValues(), fieldType);

    auto &fieldLayout = getOrCalculate(fieldType);
    // If we cannot calculate the layout of a field, we bail because reporting
    // an incorrect size for this struct might result in miscalculating the
    // locations of neighboring variables.
    if (!fieldLayout)
      return {};

    uint64_t fieldOffset = llvm::alignTo(size, fieldLayout->getAlignment());
    layout.addField({fieldOffset, fieldLayout->getByteSize(),
                     fieldLayout->getAlignment(),
                     fieldType.getAsVoidPointer()});
    size = fieldOffset + fieldLayout->getByteSize();
    align = std::max(align, fieldLayout->getAlignment());
  }
  layout.setByteSize(llvm::alignTo(size, align));
  layout.setAlignment(align);
  return layout;
}

const std::optional<MojoTypeDataLayout> &
MojoTypeDataLayoutContext::Impl::getOrCalculate(MojoASTTypeRef type) {
  auto it = cache.find(type.getAsVoidPointer());
  if (it != cache.end())
    return it->second;
  auto ret = cache.try_emplace(type.getAsVoidPointer(), calculate(type));
  return ret.first->second;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculate(MojoASTTypeRef type) {
  if (MojoASTDeclRef declRef = context.getDecl(type)) {
    if (LIT::StructDeclOp structDeclOp =
            dyn_cast_if_present<LIT::StructDeclOp>(declRef.getIfOperation())) {
      return calculateForStruct(type, structDeclOp);
    }
  }
  std::optional<uint64_t> size =
      DataLayoutInterface::getTypeStoreSize(targetInfo, type.getMLIRType());
  std::optional<uint64_t> alignment =
      DataLayoutInterface::getTypeABIAlign(targetInfo, type.getMLIRType());
  if (!size || !alignment)
    return {};

  return MojoTypeDataLayout(*size, *alignment);
}

//===----------------------------------------------------------------------===//
// MojoTypeDataLayoutContext
//===----------------------------------------------------------------------===//

MojoTypeDataLayoutContext::~MojoTypeDataLayoutContext() = default;

MojoTypeDataLayoutContext::MojoTypeDataLayoutContext(MojoParserContext &context,
                                                     TargetInfoAttr targetInfo)
    : impl(std::make_unique<MojoTypeDataLayoutContext::Impl>(context,
                                                             targetInfo)) {}

const std::optional<MojoTypeDataLayout> &
MojoTypeDataLayoutContext::getOrCalculate(MojoASTTypeRef type) {
  return impl->getOrCalculate(type);
}
