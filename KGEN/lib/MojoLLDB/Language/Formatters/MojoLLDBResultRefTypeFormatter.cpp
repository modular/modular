//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLLDBResultRefTypeFormatter.h"
#include "../../Logging/Errors.h"
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
  auto findEffectiveValue = [&]() -> ValueObjectSP {
    auto nonSyntheticValue = valobj.GetNonSyntheticValue();
    if (!nonSyntheticValue)
      return {};
    auto pointerValue = nonSyntheticValue->GetChildAtIndex(0);
    if (!pointerValue)
      return {};
    auto implValue = pointerValue->GetChildAtIndex(0);
    if (!implValue)
      return {};
    return implValue->HasSyntheticValue() ? implValue->GetSyntheticValue()
                                          : implValue;
  };

  ValueObjectSP effectiveValue = findEffectiveValue();
  if (!effectiveValue) {
    EMIT_BUG_REPORT_MESSAGE(
        "unable to inspect the REPL resultant variable '{0}'.",
        valobj.GetTypeName().GetCString());
    return false;
  }

  std::string dest;
  effectiveValue->GetSummaryAsCString(dest, summaryOptions);
  if (dest.empty())
    return false;
  stream << dest;
  return true;
}
