//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for stability markers (@stable decorator).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_STABILITYMARKERS_H
#define KGEN_MOJOPARSER_STABILITYMARKERS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN::LIT {

/// Returns true if the given package name has opted into stability tracking.
/// Packages that opt in treat symbols without @stable as unstable APIs.
/// Currently only "std" is opted in; "test_std_mock" is a test-only stand-in.
///
/// This is the single source of truth for the opted-in set. The same check
/// is used by the compiler (for unstable-API warnings) and by the doc
/// generator (to decide whether to show stability labels).
bool isPackageOptedIntoStabilityMarkers(llvm::StringRef packageName);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_STABILITYMARKERS_H
