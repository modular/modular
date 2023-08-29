//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H

#include "lldb/Target/Language.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <vector>

namespace lldb_private {
class MojoLanguage : public lldb_private::Language {

public:
  MojoLanguage() = default;

  ~MojoLanguage() override = default;

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeMojo;
  }

  std::unique_ptr<TypeScavenger> GetTypeScavenger() override { return nullptr; }
  lldb::TypeCategoryImplSP GetFormatters() override;

  HardcodedFormatters::HardcodedSummaryFinder GetHardcodedSummaries() override;

  HardcodedFormatters::HardcodedSyntheticFinder
  GetHardcodedSynthetics() override;

  bool IsNilReference(ValueObject &valobj) override {
    if (!(valobj.GetObjectRuntimeLanguage() == lldb::eLanguageTypeMojo) ||
        !valobj.IsPointerType()) {
      return false;
    }
    bool canReadValue = true;
    bool isZero = valobj.GetValueAsUnsigned(0, &canReadValue) == 0;
    return canReadValue && isZero;
  }

  llvm::StringRef GetNilReferenceSummaryString() override { return "None"; }

  bool IsSourceFile(llvm::StringRef filePath) const override {
    return filePath.ends_with(".mojo") || filePath.ends_with("🔥");
  }

  const Highlighter *GetHighlighter() const override { return nullptr; }

  //===--------------------------------------------------------------------===//
  // Static Functions
  //===--------------------------------------------------------------------===//

  static void Initialize();

  static void Terminate();

  static lldb_private::Language *CreateInstance(lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "mojo"; }

  bool SymbolNameFitsToLanguage(Mangled mangled) const override { return true; }

  bool DemangledNameContainsPath(llvm::StringRef path,
                                 ConstString demangled) const override {
    return false;
  }

  ConstString
  GetDemangledFunctionNameWithoutArguments(Mangled mangled) const override {
    return {};
  }

  bool GetFunctionDisplayName(const SymbolContext *sc,
                              const ExecutionContext *exe_ctx,
                              FunctionNameRepresentation representation,
                              Stream &s) override {
    return false;
  }

  std::vector<ConstString>
  GenerateAlternateFunctionManglings(const ConstString mangled) const override {
    return {};
  }

  ConstString FindBestAlternateFunctionMangledName(
      const Mangled mangled, const SymbolContext &sym_ctx) const override {
    return {};
  }

  llvm::StringRef GetInstanceVariableName() override { return "self"; }

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGE_H
