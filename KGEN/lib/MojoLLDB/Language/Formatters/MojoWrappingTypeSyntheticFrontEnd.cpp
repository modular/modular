//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoWrappingTypeSyntheticFrontEnd.h"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

size_t MojoWrappingTypeSyntheticFrontEnd::CalculateNumChildren() {
  if (lldb::ValueObjectSP value = getEffectiveValue())
    return value->GetNumChildren();
  return 0;
}

lldb::ValueObjectSP
MojoWrappingTypeSyntheticFrontEnd::GetChildAtIndex(size_t idx) {
  if (lldb::ValueObjectSP value = getEffectiveValue())
    return value->GetChildAtIndex(idx, /*can_create=*/true);
  return {};
}

size_t
MojoWrappingTypeSyntheticFrontEnd::GetIndexOfChildWithName(ConstString name) {
  if (lldb::ValueObjectSP value = getEffectiveValue())
    return value->GetIndexOfChildWithName(name);
  return 0;
}

bool MojoWrappingTypeSyntheticFrontEnd::Update() {
  wrappedValue = m_backend.GetSP();
  for (size_t pos : path) {
    if (wrappedValue && wrappedValue->MightHaveChildren() &&
        wrappedValue->GetNumChildren() > pos)
      wrappedValue = wrappedValue->GetChildAtIndex(pos, /*can_create=*/true);
    else
      wrappedValue = {};
  }
  // We don't emit any user facing errors here because they are taken care of
  // by `mojoREPLResultRefTypeSummaryProvider`.
  return false;
}

lldb::ValueObjectSP MojoWrappingTypeSyntheticFrontEnd::GetSyntheticValue() {
  if (wrappedValue)
    return wrappedValue->GetSyntheticValue();
  return {};
}

lldb::ValueObjectSP MojoWrappingTypeSyntheticFrontEnd::getEffectiveValue() {
  if (auto synthetic = GetSyntheticValue())
    return synthetic;
  return wrappedValue;
}

ConstString MojoWrappingTypeSyntheticFrontEnd::GetSyntheticTypeName() {
  if (auto synthetic = GetSyntheticValue())
    return synthetic->GetDisplayTypeName();
  return {};
}

bool MojoWrappingTypeSyntheticFrontEnd::MightHaveChildren() {
  if (lldb::ValueObjectSP value = getEffectiveValue())
    return value->MightHaveChildren();
  return false;
}
