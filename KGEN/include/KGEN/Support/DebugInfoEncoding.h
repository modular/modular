//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares logic for how KGEN entities are encoded in DebugInfo,
// providing shared utilites between the compiler and debugger.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DEBUGINFOENCODING_H
#define KGEN_DEBUGINFOENCODING_H

#include "KGEN/KGENDialect/KGENDType.h"

namespace M::KGEN::DebugInfoEncoding {

//===----------------------------------------------------------------------===//
// KGENDType Encoding
//===----------------------------------------------------------------------===//

/// Get the fully qualified name used for a KGENDType in debuginfo.
std::string getKGENDTypeAsString(KGENDType dtype);
/// Get a KGENDType from a fully qualified name in debuginfo.
FailureOr<KGENDType> getKGENDTypeFromString(StringRef str);

} // namespace M::KGEN::DebugInfoEncoding

#endif // KGEN_DEBUGINFOENCODING_H
