//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_GENMARKDOWN_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_GENMARKDOWN_H

namespace M {

class BackendRegistry;

/// Registers the "gen-markdown" backend.
void registerGenMarkdownBackend(BackendRegistry &registry);
} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_GENMARKDOWN_H
