//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLLDBResultRefTypeFormatter.h"
#include "MojoWrappingTypeSyntheticFrontEnd.h"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoREPLResultRefTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  return new MojoWrappingTypeSyntheticFrontEnd(*valobjSP, {0, 0});
}

bool M::KGEN::Mojo::mojoREPLResultRefTypeSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summaryOptions) {
  auto nonSyntheticValue = valobj.GetNonSyntheticValue();
  if (!nonSyntheticValue)
    return false;
  auto pointerValue = nonSyntheticValue->GetChildAtIndex(0);
  if (!pointerValue)
    return false;
  auto implValue = pointerValue->GetChildAtIndex(0);
  if (!implValue)
    return false;

  ValueObjectSP effectiveValue = implValue->HasSyntheticValue()
                                     ? implValue->GetSyntheticValue()
                                     : implValue;
  std::string dest;
  effectiveValue->GetSummaryAsCString(dest, summaryOptions);
  if (dest.empty())
    return false;
  stream << dest;
  return true;
}
