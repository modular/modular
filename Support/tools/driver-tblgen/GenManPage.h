//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_GENMANPAGE_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_GENMANPAGE_H

namespace M {

class BackendRegistry;

/// Registers the "gen-man-page" backend.
void registerGenManPageBackend(BackendRegistry &registry);
} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_GENMANPAGE_H
