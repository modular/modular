//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJOTBLGEN_GENHELPTEXT_H
#define KGEN_TOOLS_MOJOTBLGEN_GENHELPTEXT_H

namespace M {

class BackendRegistry;

/// Registers the "gen-help-text" backend.
void registerGenHelpTextBackend(BackendRegistry &registry);
} // namespace M

#endif // KGEN_TOOLS_MOJOTBLGEN_GENHELPTEXT_H
