//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_CONSTANTS_H
#define KGEN_SUPPORT_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN {

/// On-disk directory name (relative to the modular cache root) where the
/// Mojo compile cache lives.
inline constexpr llvm::StringLiteral kMojoCacheBaseDirName = ".mojo_cache";

} // namespace M::KGEN

#endif // KGEN_SUPPORT_CONSTANTS_H
