//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/DebugInfoEncoding.h"
#include "KGEN/KGENDialect/KGENDType.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KGENDType Encoding
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
