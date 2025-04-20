//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLanguage.h"
#include "../Utils/Errors.h"
#include "Formatters/MojoDecoratorBasedTypeFormatter.h"
#include "Formatters/MojoKGENVariantTypeFormatter.h"
#include "Formatters/MojoListTypeFormatter.h"
#include "lldb/API/SBValue.h"
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

static bool
builtinStringSummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &summaryOptions) {
  // If we fail to read the string, we show some placeholder error text.
  // Otherwise, if we return false, for example, LLDB would print the contents
  // of the inner List.
  auto onError = [&stream](llvm::StringRef why = {}) {
    stream << "Summary Unavailable" << why;
    return true;
  };

  ValueObjectSP sizeField = valobj.GetChildMemberWithName("_len");
  if (!sizeField || !sizeField->GetError().Success())
    return onError("; could not find String._len field");

  // The REPL sees a struct around an int, but DWARF shows directly the int.
  lldb::ValueObjectSP sizeVal =
      sizeField->IsScalarType() ? sizeField
                                : sizeField->GetChildMemberWithName("value");
  if (!sizeVal || !sizeVal->GetError().Success())
    return onError();

  bool success = true;
  size_t size = sizeVal->GetValueAsUnsigned(0, &success);
  if (!success)
    return onError();

  // If the size is 0, the data address might be invalid.
  if (size == 0) {
    stream << "\"\"";
    return true;
  }

  // Otherwise, decode the pointer.
  ValueObjectSP dataVal = valobj.GetChildMemberWithName("_data");
  if (!dataVal || !dataVal->GetError().Success())
    return onError("; could not find String._data field");

  // The REPL sees a struct around a pointer, but DWARF shows directly the
  // pointer.
  ValueObjectSP dataPointer = dataVal->IsPointerType()
                                  ? dataVal
                                  : dataVal->GetChildMemberWithName("address");

  if (!dataPointer || !dataPointer->GetError().Success())
    return onError();

  lldb::addr_t data = dataPointer->GetPointerValue();
  if (!data || data == LLDB_INVALID_ADDRESS)
    return onError();

  // Now that we have the data pointer, dereference to read the string data.
  StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);

  if (summaryOptions.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped) {
    size_t maxSize = valobj.GetTargetSP()->GetMaximumSizeOfStringSummary();
    if (size > maxSize) {
      size = maxSize;
      options.SetIsTruncated(true);
    }
  }

  DataExtractor extractor;
  const size_t bytesRead = dataPointer->GetPointeeData(extractor, 0, size);
  if (bytesRead < size)
    return onError();

  options.SetData(std::move(extractor));
  options.SetStream(&stream);
  options.SetPrefixToken(nullptr);
  options.SetQuote('"');
  options.SetSourceSize(size);
  options.SetBinaryZeroIsTerminator(true);
  return StringPrinter::ReadBufferAndDumpToStream<
      StringPrinter::StringElementType::ASCII>(options);
}

/// The None type is rendered nicely if its summary is the "None" string.
static bool kgenNoneSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &summaryOptions) {
  stream << "None";
  return true;
}

/// Bool types are rendered nicely as True or False.
static bool boolSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &summaryOptions) {
  bool success = false;
  int val = valobj.GetValueAsUnsigned(/*default=*/0, &success);
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
  auto numChildren = getExpectedValueOr(valobj.GetNumChildren(), 0u);
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

  constexpr const char *kListRegex =
      R"(^!lit.struct<@stdlib::@collections::@list::@"?List[\[<].*)";
  constexpr const char *kLLDBFormatterWrappingTypeRegex =
      R"(.* {@stdlib::utils::_visualizers::lldb_formatter_wrapping_type\(.*)";

  // Formatters are matched in reverse order.
  AddCXXSynthetic(mojoCategorySP,
                  MojoLLDBWrappingTypeTypeSyntheticFrontEndCreator,
                  "Mojo decorator-based synthetic children",
                  kLLDBFormatterWrappingTypeRegex, synthFlags,
                  /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP, mojoListSyntheticFrontEndCreator,
                  "Mojo List synthetic children", kListRegex, synthFlags,
                  /*regex=*/true);
  AddCXXSynthetic(mojoCategorySP, mojoKGENVariantSyntheticFrontEndCreator,
                  "Mojo !kgen.variant synthetic children",
                  R"(^!kgen\.variant<.*>)", synthFlags, /*regex=*/true);

  TypeSummaryImpl::Flags summaryFlags;
  summaryFlags.SetCascades(true)
      .SetSkipPointers(false)
      .SetSkipReferences(false)
      .SetDontShowChildren(false)
      .SetDontShowValue(true)
      .SetShowMembersOneLiner(false)
      .SetHideItemNames(false);

  // Summary providers are matched in reverse order.
  AddCXXSummary(mojoCategorySP, MojoLLDBWrappingTypeSummaryProvider,
                "Mojo decorator-based summary provider",
                kLLDBFormatterWrappingTypeRegex, summaryFlags, /*regex=*/true);

  summaryFlags.SetDontShowChildren(true);
  AddCXXSummary(mojoCategorySP, kgenNoneSummaryProvider,
                "!kgen.none summary provider", "!kgen.none", summaryFlags,
                /*regex=*/false);
  AddCXXSummary(mojoCategorySP, boolSummaryProvider, "bool summary provider",
                "i1", summaryFlags, /*regex=*/false);
  AddCXXSummary(
      mojoCategorySP, builtinStringSummaryProvider,
      "collections::string::string::String summary provider",
      R"(!lit.struct<(@stdlib::)?@collections::@string::@string::@String>)",
      summaryFlags, /*regex=*/true);

  summaryFlags.SetDontShowChildren(false);
  summaryFlags.SetDontShowValue(false);
  AddCXXSummary(mojoCategorySP, vectorLikeSummaryProvider,
                "collections::list::List summary provider", kListRegex,
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
