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

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::MojoDecoratorBasedTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *x, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  CompilerType type = valobjSP->GetCompilerType();
  if (!type.IsValid())
    return nullptr;
  std::shared_ptr<MojoTypeSystem> mojoTypeSystem =
      type.GetTypeSystem().dyn_cast_or_null<MojoTypeSystem>();
  if (!mojoTypeSystem)
    return nullptr;

  for (TypedAttr decorator :
       mojoTypeSystem->GetStructDecorators(type.GetOpaqueQualType())) {
    if (auto constantSymbol = dyn_cast<KGEN::SymbolConstantAttr>(decorator)) {
      SymbolRefAttr symbol = constantSymbol.getSymbol();
      auto nestedReferences = symbol.getNestedReferences();
      if (nestedReferences.size() == 2 &&
          symbol.getRootReference() == "$utils" &&
          nestedReferences[0].getValue() == "$lldb" &&
          nestedReferences[1].getValue() == "lldb_formatter_wrapping_type()") {
        return new MojoWrappingTypeSyntheticFrontEnd(*valobjSP);
      }
    }
  }
  return nullptr;
}
