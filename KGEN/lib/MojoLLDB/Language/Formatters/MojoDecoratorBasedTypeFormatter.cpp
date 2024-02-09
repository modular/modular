//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoDecoratorBasedTypeFormatter.h"
#include "../../TypeSystem/MojoTypeSystem.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "MojoWrappingTypeSyntheticFrontEnd.h"

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
    if (!m_backend.MightHaveChildren() || m_backend.GetNumChildren() == 0)
      return {};
    return m_backend.GetChildAtIndex(0, /*can_create=*/true);
  }
};
} // namespace

/// Check if the type of the given value is using the decorator
/// @lldb_formatter_wrapping_type.
static bool isUsingLLDBFormatterWrappingType(const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return false;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return false;
  std::shared_ptr<MojoTypeSystem> mojoTypeSystem =
      type.GetTypeSystem().dyn_cast_or_null<MojoTypeSystem>();
  if (!mojoTypeSystem)
    return false;

  for (TypedAttr decorator :
       mojoTypeSystem->getStructDecorators(type.GetOpaqueQualType())) {
    if (auto constantSymbol = dyn_cast<KGEN::SymbolConstantAttr>(decorator)) {
      SymbolRefAttr symbol = constantSymbol.getSymbol();
      auto nestedReferences = symbol.getNestedReferences();
      if (nestedReferences.size() == 3 &&
          symbol.getRootReference() == "stdlib" &&
          nestedReferences[0].getValue() == "debug" &&
          nestedReferences[1].getValue() == "visualizers" &&
          nestedReferences[2].getValue() == "lldb_formatter_wrapping_type()") {
        return true;
      }
    }
  }
  return false;
}

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::mojoDecoratorBasedTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *x, const ValueObjectSP &valobjSP) {
  if (isUsingLLDBFormatterWrappingType(valobjSP))
    return new MojoWrappingTypeSyntheticFrontEnd(*valobjSP);
  return nullptr;
}

bool M::KGEN::Mojo::mojoDecoratorBasedSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &summaryOptions) {
  ValueObjectSP nonSyntheticValobj = valobj.GetNonSyntheticValue();
  if (!isUsingLLDBFormatterWrappingType(nonSyntheticValobj))
    return false;
  ValueObjectSP impl = nonSyntheticValobj->GetChildAtIndex(0);
  std::string dest;
  impl->GetSummaryAsCString(dest, summaryOptions);
  if (dest.empty())
    return false;
  stream << dest;
  return true;
}
