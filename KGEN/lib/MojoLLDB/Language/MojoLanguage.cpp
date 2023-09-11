//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLanguage.h"
#include "MojoDynamicVectorTypeFormatter.h"
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
  return new MojoLanguage();
}

static void LoadLibMojoFormatters(lldb::TypeCategoryImplSP mojoCategorySP) {
  if (!mojoCategorySP)
    return;

  SyntheticChildren::Flags synthFlags;
  synthFlags.SetCascades(true).SetSkipPointers(true).SetSkipReferences(true);
  AddCXXSynthetic(
      mojoCategorySP, MojoDynamicVectorSyntheticFrontEndCreator,
      "mojo DynamicVector synthetic children",
      "^!kgen.declref<@\"\\$utils\"::@\"\\$vector\"::@DynamicVector<.*>>$",
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
