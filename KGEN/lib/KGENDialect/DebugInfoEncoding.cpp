//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/DebugInfoEncoding.h"
#include "KGEN/KGENDialect/KGENDType.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KGENDType Native Encoding
//===----------------------------------------------------------------------===//
constexpr StringLiteral KGENDTypeQualifiedPrefix = "kgen.dtype.";

std::string DebugInfoEncoding::getKGENDTypeAsString(KGENDType dtype) {
  return KGENDTypeQualifiedPrefix.str() + dtype.getAsString();
}

FailureOr<KGENDType> DebugInfoEncoding::getKGENDTypeFromString(StringRef str) {
  if (!str.starts_with(KGENDTypeQualifiedPrefix))
    return failure();

  return KGENDType::getFromString(
      str.drop_front(KGENDTypeQualifiedPrefix.size()));
}

//===----------------------------------------------------------------------===//
// KGENDType C++ Encoding
//===----------------------------------------------------------------------===//

std::optional<std::string>
DebugInfoEncoding::getKGENDTypeAsCppString(KGENDType dtype) {
  if (dtype.isBool())
    return "bool";

  if (dtype.isAddress())
    return "void *";

  if (dtype.isInt()) {
    size_t width = dtype.getIntegerWidthInBits();
    if (width < 8 || width > 64)
      return {};

    std::string name;
    llvm::raw_string_ostream oss(name);
    if (!dtype.isSInt())
      oss << 'u';
    oss << "int" << dtype.getIntegerWidthInBits() << "_t";
    return name;
  }

  if (dtype.isFloat()) {
    switch (dtype.getWidthInBits()) {
    case 32:
      return "float";
    case 64:
      return "double";
    }
  }

  return {};
}
