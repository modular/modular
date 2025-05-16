//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/ManglingUtils.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;

void KGEN::prettyPrintParameter(TypedAttr value, raw_ostream &os) {
  if (auto typeCst = dyn_cast<TypeParamAttr>(value)) {
    // Pretty print common type values.
    Type typeValue = typeCst.getTypeValue();
    if (auto structInst = dyn_cast<StructInstanceType>(typeValue)) {
      // Print full struct type name with its parameters recursively.
      os << structInst.getName().strref();
      if (!structInst.getParamValues().empty()) {
        os << '[';
        llvm::interleave(
            structInst.getParamValues(), os,
            [&](TypedAttr paramValue) { prettyPrintParameter(paramValue, os); },
            ",");
        os << ']';
      }
    } else if (auto typeValueType = dyn_cast<TypeValueType>(typeValue)) {
      // Print the wrapped type parameter.
      prettyPrintParameter(typeValueType.getTypeValue(), os);
    } else {
      os << getParamAsString(value);
    }
    return;
  }

  if (auto genref = dyn_cast<TypeGeneratorRefAttr>(value)) {
    // Print type symbol references with its name and its parameters
    // recursively.
    os << genref.getSymbol().getLeafReference().strref();
    if (!genref.getParamValues().empty()) {
      os << '[';
      llvm::interleave(
          genref.getParamValues(), os,
          [&](TypedAttr paramValue) { prettyPrintParameter(paramValue, os); },
          ",");
      os << ']';
    }
    return;
  }

  if (auto typeInstanceRef = dyn_cast<TypeInstanceRefAttr>(value)) {
    os << typeInstanceRef.getSymbol().getLeafReference().strref();
    return;
  }

  // Fallback to default format.
  os << getParamAsString(value);
}

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// Return a string that is a unique specification of the specified parameter.
static void printParameterMangling(TypedAttr value, raw_ostream &os) {
  // TODO: Don't print #lit.any.origin, they are singletons!
  // if (isa<AnyOriginAttr>(value))
  //  return;

  // Handle VariadicAttr of types as well, which are common for variadic packs.
  if (auto variadic = dyn_cast<VariadicAttr>(value)) {
    os << '[';
    llvm::interleaveComma(variadic.getValues(), os, [&](TypedAttr paramValue) {
      printParameterMangling(paramValue, os);
    });
    os << ']';
    return;
  }

  // Print SymbolRefAttr without a leading @, because that antagonizes ELF and
  // isn't required to disambiguate symbols.
  auto result = getParamAsString(value);
  StringRef resultToPrint = result;
  if (resultToPrint.starts_with("@"))
    resultToPrint = resultToPrint.drop_front();

  // The kgen representation will always be a valid choice.
  os << resultToPrint;
}

std::string
KGEN::mangleParameterValues(GeneratorOpInterface generator,
                            ArrayRef<TypedAttr> inputParamValues,
                            function_ref<std::string(StringRef)> getPrefix) {
  Builder b(generator.getContext());
  std::string prefix = getPrefix(generator.getName());

  if (inputParamValues.empty())
    return prefix + generator.getName().str();

  std::string result;
  llvm::raw_string_ostream os(result);
  os << prefix;
  os << generator.getName();

  // Mangle in things like "size=42" for each of the parameters to make it easy
  // to read the resultant symbol and also to make things unique when
  // instantiated with different values.
  auto inputParamDecls = generator.getInputParams();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterMangling(value, os);
  }

  // Having "@" in mangled names is invalid for ELF files and triggers error at
  // linking stage, so replace them.  We replace "@" with "\eA" and "\e" with
  // "\e\e" to make sure there are no collisions. \e is a non-standard C++
  // extension so we use \033 for portability.
  for (size_t pos = result.find_first_of("@\033"); pos != std::string::npos;
       pos = result.find_first_of("@\033", pos + 2)) {
    result.replace(pos, 1, result[pos] == '@' ? "\033" : "\033\033");
  }
  return result;
}
