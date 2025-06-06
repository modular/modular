//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares logic for how KGEN entities are encoded in DebugInfo,
// providing shared utilities between the compiler and debugger.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_DEBUGINFOENCODING_H
#define KGEN_DEBUGINFOENCODING_H

#include "KGEN/KGENDialect/KGENDType.h"

namespace M::KGEN::DebugInfoEncoding {

//===----------------------------------------------------------------------===//
// KGENDType Native Encoding
//===----------------------------------------------------------------------===//

/// Get the fully qualified name used for a KGENDType in debuginfo.
std::string getKGENDTypeAsString(KGENDType dtype);
/// Get a KGENDType from a fully qualified name in debuginfo.
FailureOr<KGENDType> getKGENDTypeFromString(StringRef str);

//===----------------------------------------------------------------------===//
// KGENDType C++ Encoding
//===----------------------------------------------------------------------===//

/// Get the equivalent C++ type name for a KGENDType if supported by C++.
/// Otherwise returns nullopt.
std::optional<std::string> getKGENDTypeAsCppString(KGENDType dtype);

} // namespace M::KGEN::DebugInfoEncoding

#endif // KGEN_DEBUGINFOENCODING_H
