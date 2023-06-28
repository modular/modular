//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJOTBLGEN_GENMANPAGE_H
#define KGEN_TOOLS_MOJOTBLGEN_GENMANPAGE_H

namespace M {

class BackendRegistry;

/// Registers the "gen-man-page" backend.
void registerGenManPageBackend(BackendRegistry &registry);
} // namespace M

#endif // KGEN_TOOLS_MOJOTBLGEN_GENMANPAGE_H
