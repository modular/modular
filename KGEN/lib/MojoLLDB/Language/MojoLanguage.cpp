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

static void
LoadLibMojoFormatters(const lldb::TypeCategoryImplSP &mojoCategorySP) {
  if (!mojoCategorySP)
    return;

  SyntheticChildren::Flags synthFlags;
  synthFlags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(true);
  // Formatters are matched in reverse order, so this one that uses .* should
  // be added first so that it's the last one to be matched against. In fact,
  // this formatter acts like a sink that will match everything that doesn't
  // match the other formatters.
  AddCXXSynthetic(mojoCategorySP,
                  MojoDecoratorBasedTypeSyntheticFrontEndCreator,
                  "Mojo decorator-based synthetic children", ".*", synthFlags,
                  /*regex=*/true);
  // This formatter will replace a REPLResultRef with the value of its inner
  // type. REPLResultRef owns a pointer, so we need to dereference the pointer.
  AddCXXSynthetic(mojoCategorySP, MojoREPLResultRefTypeSyntheticFrontEndCreator,
                  "REPLResultRefType synthetic children",
                  "^!lit.replresultref<.*>$", synthFlags, /*regex=*/true);
  AddCXXSynthetic(
      mojoCategorySP, MojoDynamicVectorSyntheticFrontEndCreator,
      "Mojo DynamicVector synthetic children",
      R"(^!kgen.declref<@"\$utils"::@"\$vector"::@DynamicVector<.*>>$)",
      synthFlags, /*regex=*/true);
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
