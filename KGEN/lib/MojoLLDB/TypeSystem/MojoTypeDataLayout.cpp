//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoTypeDataLayout.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "Support/LLVMAlignToMacro.h"

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

  /// Calculate the data layout of the given pack.
  std::optional<MojoTypeDataLayout> calculateForPack(MojoASTTypeRef typeRef,
                                                     PackType packType);

  /// Calculate the data layout of a struct-like collection of types.
  std::optional<MojoTypeDataLayout>
  calculateForStructLike(ArrayRef<MojoASTTypeRef> fieldTypes);

  DenseMap<Type, std::optional<MojoTypeDataLayout>> cache;
  MojoParserContext &context;
  TargetInfoAttr targetInfo;
};

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStructLike(
    ArrayRef<MojoASTTypeRef> fieldTypes) {
  uint64_t size = 0, align = 1;
  MojoTypeDataLayout layout;

  for (MojoASTTypeRef fieldType : fieldTypes) {
    auto &fieldLayout = getOrCalculate(fieldType);
    // If we cannot calculate the layout of a field, we bail because reporting
    // an incorrect size for this struct might result in miscalculating the
    // locations of neighboring variables.
    if (!fieldLayout)
      return {};
    uint64_t fieldOffset;
    CHECKED_LLVM_ALIGN_TO(fieldOffset, size, fieldLayout->getAlignment());
    layout.addField({fieldOffset, fieldLayout->getByteSize(),
                     fieldLayout->getAlignment(), fieldType});
    size = fieldOffset + fieldLayout->getByteSize();
    align = std::max(align, fieldLayout->getAlignment());
  }
  layout.setByteSize(llvm::alignTo(size, align));
  layout.setAlignment(align);
  return layout;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForStruct(
    MojoASTTypeRef typeRef, LIT::StructDeclOp structOp) {
  auto refType = cast<DeclRefType>(typeRef);
  return calculateForStructLike(llvm::map_to_vector(
      structOp.getFieldDecls(), [&](LIT::StructFieldOp field) {
        MojoASTTypeRef fieldType =
            MojoASTTypeRef(field.getTypeAttr().getValue());
        // If the DeclRefType has parameters, try to evaluate and substitute
        // them into the type.
        if (!refType.getParamValues().empty()) {
          fieldType = context.concretizeType(typeRef, refType.getParamValues(),
                                             fieldType);
        }
        return fieldType;
      }));
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculateForPack(MojoASTTypeRef typeRef,
                                                  PackType packType) {
  auto attr = dyn_cast<VariadicAttr>(packType.getVariadic());
  if (!attr)
    return {};
  return calculateForStructLike(
      llvm::map_to_vector(attr.getValues(), [&](TypedAttr value) {
        return MojoASTTypeRef(value);
      }));
}

const std::optional<MojoTypeDataLayout> &
MojoTypeDataLayoutContext::Impl::getOrCalculate(MojoASTTypeRef type) {
  auto it = cache.find(type.getMLIRType());
  if (it != cache.end())
    return it->second;
  auto ret = cache.try_emplace(type.getMLIRType(), calculate(type));
  return ret.first->second;
}

std::optional<MojoTypeDataLayout>
MojoTypeDataLayoutContext::Impl::calculate(MojoASTTypeRef type) {
  // A REPLResultRefType is effectively a pointer, so we transform it into one.
  if (auto replType = dyn_cast<LIT::REPLResultRefType>(type))
    return calculate(KGEN::PointerType::get(replType.getElementType()));

  if (MojoASTDeclRef declRef = context.getDecl(type)) {
    if (LIT::StructDeclOp structDeclOp =
            dyn_cast_if_present<LIT::StructDeclOp>(declRef.getIfOperation())) {
      return calculateForStruct(type, structDeclOp);
    }
  }

  if (auto packType = dyn_cast<KGEN::PackType>(type))
    return calculateForPack(type, packType);

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
