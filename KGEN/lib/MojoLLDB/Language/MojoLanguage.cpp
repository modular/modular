//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLanguage.h"
#include "Formatters/MojoDecoratorBasedTypeFormatter.h"
#include "Formatters/MojoDynamicVectorTypeFormatter.h"
#include "Formatters/MojoLLDBResultRefTypeFormatter.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/VectorType.h"
#include <lldb/API/SBValue.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace M::KGEN::Mojo;

LLDB_PLUGIN_DEFINE(MojoLanguage)

void MojoLanguage::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Mojo Language",
                                CreateInstance);
}

void MojoLanguage::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

//===----------------------------------------------------------------------===//
// Static Functions
//===----------------------------------------------------------------------===//

Language *MojoLanguage::CreateInstance(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeMojo:
    return new MojoLanguage();
  default:
    return nullptr;
  }
}

/// The None type is rendered nicely if its summary is the "None" string.
static bool kgenNoneSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &summaryOptions) {
  stream << "None";
  return true;
}

// Bool types are rendered nicely as True or False.
static bool
popScalarBoolSummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &summaryOptions) {
  ValueObjectSP valueChild(valobj.GetChildAtIndex(0, true));
  if (!valueChild)
    return false;
  bool success = false;
  int val = valueChild->GetValueAsUnsigned(/*default=*/0, &success);
  if (success) {
    if (val == 0)
      stream << "False";
    else
      stream << "True";
    return true;
  }
  return false;
}

/// Create a short summary for a vector-like object. It includes the size of the
/// container and, if the children are scalars or have summaries of their own,
/// they will be displayed as well.
static bool
vectorLikeSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &summaryOptions) {
  size_t numChildren = valobj.GetNumChildren();
  stream.Format("(size {0})", numChildren);

  // We'll limit the amount of characters to use when displaying children.
  // In practice we can go beyond this limit by a few characters.
  const size_t maxChildrenSummaryLength = 32;
  std::string childrenSummary = "[";

  size_t i = 0;
  for (; i < numChildren; ++i) {
    // If we exceeded the number of characters, we break;
    if (childrenSummary.size() > maxChildrenSummaryLength)
      break;

    ValueObjectSP child = valobj.GetChildAtIndex(i);

    llvm::StringRef childText;
    std::string childSummary;
    child->GetSummaryAsCString(childSummary, summaryOptions);
    if (childSummary.empty())
      childText = child->GetValueAsCString();
    else
      childText = childSummary;

    // If we can't generate some text for the current child, we stop.
    if (childText.empty())
      break;

    if (i > 0)
      childrenSummary += ", ";
    childrenSummary += childText;
  }

  // If we printed some children, we include them in the output stream.
  if (i > 0) {
    // If we stopped early, we add `...` to show that there are more elements.
    if (i < numChildren)
      childrenSummary += ", ...";

    childrenSummary += "]";
    stream << childrenSummary;
  }
  return true;
}

static void
LoadLibMojoFormatters(const lldb::TypeCategoryImplSP &mojoCategorySP) {
  if (!mojoCategorySP)
    return;

  // These settings are the same as the C++ ones.
  SyntheticChildren::Flags synthFlags;
  synthFlags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(true);

  // Formatters are matched in reverse order, so this one that uses .* should
  // be added first so that it's the last one to be matched against. In fact,
  // this formatter acts like a sink that will match everything that doesn't
  // match the other formatters.
  AddCXXSynthetic(mojoCategorySP,
                  mojoDecoratorBasedTypeSyntheticFrontEndCreator,
                  "Mojo decorator-based synthetic children", ".*", synthFlags,
                  /*regex=*/true);
  // This formatter will replace a REPLResultRef with the value of its inner
  // type. REPLResultRef owns a pointer, so we need to dereference the pointer.
  AddCXXSynthetic(mojoCategorySP, mojoREPLResultRefTypeSyntheticFrontEndCreator,
                  "REPLResultRefType synthetic children",
                  "^!lit.replresultref<.*>$", synthFlags, /*regex=*/true);
  AddCXXSynthetic(
      mojoCategorySP, mojoDynamicVectorSyntheticFrontEndCreator,
      "Mojo DynamicVector synthetic children",
      R"(^!kgen.declref<@"\$stdlib"::@"\$collections"::@"\$vector"::@"?DynamicVector[\[<].*$)",
      synthFlags, /*regex=*/true);

  // These settings are the same as the C++ ones.
  TypeSummaryImpl::Flags summaryFlags;
  summaryFlags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(true)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  // Summary providers are matched in reverse order, so this one that uses .*
  // should be added first so that it's the last one to be matched against. In
  // fact, this provider acts like a sink that will match everything that
  // doesn't match the other providers.
  AddCXXSummary(mojoCategorySP, mojoDecoratorBasedSummaryProvider,
                "Mojo decorator-based summary provider", ".*", summaryFlags,
                /*regex=*/true);
  AddCXXSummary(mojoCategorySP, kgenNoneSummaryProvider,
                "!kgen.none summary provider", "!kgen.none", summaryFlags,
                /*regex=*/false);
  AddCXXSummary(mojoCategorySP, popScalarBoolSummaryProvider,
                "!pop.scalar<bool> summary provider", "!pop.scalar<bool>",
                summaryFlags, /*regex=*/false);
  AddCXXSummary(mojoCategorySP, mojoREPLResultRefTypeSummaryProvider,
                "REPLResultRefType summary provider",
                "^!lit.replresultref<.*>$", summaryFlags, /*regex=*/true);

  summaryFlags.SetDontShowChildren(false);
  summaryFlags.SetDontShowValue(false);
  // FIXME(26722): add support for this summary provider in the REPL. Right now
  // the regex only includes the DWARF version of this type.
  AddCXXSummary(
      mojoCategorySP, vectorLikeSummaryProvider,
      "$utils::vector::DynamicVector summary provider",
      R"(^!kgen.declref<@"\$stdlib"::@"\$collections"::@"\$vector"::@"DynamicVector\[.*$)",
      summaryFlags, /*regex=*/true);
}

lldb::TypeCategoryImplSP MojoLanguage::GetFormatters() {
  static llvm::once_flag initialize;
  static TypeCategoryImplSP category;

  llvm::call_once(initialize, [this]() -> void {
    DataVisualization::Categories::GetCategory(ConstString(GetPluginName()),
                                               category);
    if (category) {
      LoadLibMojoFormatters(category);
    }
  });
  return category;
}

HardcodedFormatters::HardcodedSummaryFinder
MojoLanguage::GetHardcodedSummaries() {
  return HardcodedFormatters::HardcodedSummaryFinder();
}

HardcodedFormatters::HardcodedSyntheticFinder
MojoLanguage::GetHardcodedSynthetics() {
  return HardcodedFormatters::HardcodedSyntheticFinder();
}
