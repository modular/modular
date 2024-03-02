//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoKGENVariantTypeFormatter.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

MojoKGENVariantTypeSyntheticFrontEnd::MojoKGENVariantTypeSyntheticFrontEnd(
    const lldb::ValueObjectSP &backend)
    : SyntheticChildrenFrontEnd(*backend), content() {
  if (backend)
    Update();
}

size_t MojoKGENVariantTypeSyntheticFrontEnd::CalculateNumChildren() {
  return 1;
}

lldb::ValueObjectSP
MojoKGENVariantTypeSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (idx >= 1)
    return ValueObjectSP();
  return content;
}

lldb::ChildCacheState MojoKGENVariantTypeSyntheticFrontEnd::Update() {
  std::optional<ValueObjectSP> parsed =
      MojoKGENVariantTypeSyntheticFrontEnd::parseKGENVariant(m_backend.GetSP());
  if (!parsed)
    return lldb::ChildCacheState::eRefetch;

  content = *parsed;
  return lldb::ChildCacheState::eRefetch;
}

std::optional<ValueObjectSP>
MojoKGENVariantTypeSyntheticFrontEnd::parseKGENVariant(
    lldb::ValueObjectSP valobj) {
  valobj = valobj->GetNonSyntheticValue();
  if (!valobj || !valobj->GetError().Success())
    return {};

  size_t numChildren = valobj->GetNumChildren();
  if (numChildren < 1)
    return {};

  // The discriminator is the last field.
  ValueObjectSP discrField = valobj->GetChildAtIndex(numChildren - 1);
  if (!discrField || !discrField->GetError().Success())
    return {};

  bool success = true;
  size_t discr = discrField->GetValueAsUnsigned(0, &success);
  if (!success)
    return {};

  if (discr >= numChildren - 1)
    return {};

  ValueObjectSP dataVal = valobj->GetChildAtIndex(discr);
  if (!dataVal || !dataVal->GetError().Success())
    return {};

  return dataVal;
}

bool MojoKGENVariantTypeSyntheticFrontEnd::MightHaveChildren() { return true; }

size_t MojoKGENVariantTypeSyntheticFrontEnd::GetIndexOfChildWithName(
    lldb_private::ConstString targetName) {
  if (content->GetName() == targetName)
    return 0;
  return UINT32_MAX;
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoKGENVariantSyntheticFrontEndCreator(
    CXXSyntheticChildren *, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return nullptr;
  return new M::KGEN::Mojo::MojoKGENVariantTypeSyntheticFrontEnd(valobjSP);
}
