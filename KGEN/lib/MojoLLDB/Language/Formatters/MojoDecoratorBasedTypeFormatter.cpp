//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoDecoratorBasedTypeFormatter.h"
#include "../../TypeSystem/MojoTypeSystem.h"
#include "../../Utils/Errors.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

namespace {
/// Synthetic type front end corresponding to the @lldb_formatter_wrapping_type
/// decorator. It replaces a variable with its first child.
class WrappingTypeSyntheticFrontEnd
    : public lldb_private::SyntheticValueProviderFrontEnd {
public:
  WrappingTypeSyntheticFrontEnd(lldb_private::ValueObject &backend)
      : SyntheticValueProviderFrontEnd(backend) {}

  lldb::ValueObjectSP GetSyntheticValue() override {
    if (!m_backend.MightHaveChildren() ||
        getExpectedValueOr(m_backend.GetNumChildren(), 0u) == 0)
      return {};
    return m_backend.GetChildAtIndex(0, /*can_create=*/true);
  }

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    if (!MightHaveChildren())
      return 0;
    return GetSyntheticValue()->GetNumChildren();
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
    return GetSyntheticValue()->GetChildAtIndex(idx);
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    return GetSyntheticValue()->GetIndexOfChildWithName(name);
  }

  bool MightHaveChildren() override {
    // If the summary provider for this child asks for no children, then we
    // simply report as this type has no children, otherwise structs like `Bool`
    // are displayed with its nested `i1` field.
    //
    // Exception: for multi-element types like Tuple's !kgen.pack, the
    // HideChildren flag is intended to suppress redundant children on the
    // pack itself, not to hide Tuple's elements from the wrapping display.
    lldb::ValueObjectSP sv = GetSyntheticValue();
    if (!sv)
      return false;
    lldb::TypeSummaryImplSP typeSummary = sv->GetSummaryFormat();
    if (typeSummary && (typeSummary->GetOptions() & eTypeOptionHideChildren) &&
        getExpectedValueOr(sv->GetNumChildren(), 0u) <= 1)
      return false;
    return sv->MightHaveChildren();
  }
};
} // namespace

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *x, const ValueObjectSP &valobjSP) {
  return new WrappingTypeSyntheticFrontEnd(*valobjSP);
}

bool M::KGEN::Mojo::MojoLLDBWrappingTypeSummaryProvider(
    ValueObject &valobj, Stream &stream, TypeSummaryOptions summaryOptions) {
  ValueObjectSP nonSyntheticValobj = valobj.GetNonSyntheticValue();
  ValueObjectSP impl = nonSyntheticValobj->GetChildAtIndex(0);
  if (!impl)
    return false;
  std::string dest;
  impl->GetSummaryAsCString(dest, summaryOptions);
  if (!dest.empty()) {
    stream << dest;
    return true;
  }
  // Fall back to scalar value if available (e.g. index, si8, f32).
  if (const char *val = impl->GetValueAsCString()) {
    stream << val;
    return true;
  }
  return false;
}
